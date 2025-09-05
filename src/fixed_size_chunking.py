import os
import json
import time
import numpy as np
import logging
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import JSONLoader
import tiktoken
from openai import OpenAI
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Logging Setup ---

def setup_logging(chunk_size, chunk_overlap):
    log_folder = 'logs'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_file = os.path.join(log_folder, f'chunking_size-{chunk_size}-ol-{chunk_overlap}.log')

    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler for detailed logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Create console handler for important progress and errors only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors on console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Set up root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    print(f"✓ Logging configured. Detailed logs: {log_file}")

# --- Loading and Chunking ---

def load_documents(file_path):
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata['video_id'] = record.get('video_id')
        metadata['live_number'] = record.get('live_number')
        metadata['published_at'] = record.get('published_at')
        metadata['title'] = record.get('title')
        metadata['transcript_json'] = record.get('transcript_json')
        return metadata

    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[]',
        content_key='transcript_text',
        metadata_func=metadata_func
    )
    
    documents = loader.load()
    logging.debug(f'Loaded {len(documents)} documents from {file_path}.')
    return documents

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def find_start_time_by_text(chunk_text: str, transcript_json: list, transcript_text: str, max_search_words: int = 10):
    """
    Finds the start time for a chunk by searching for its beginning text in the existing transcript_text.
    Simple and direct approach.
    """
    if not transcript_json or not chunk_text.strip() or not transcript_text:
        return None
    
    # Get the first few words of the chunk to search for
    chunk_words = chunk_text.strip().split()[:max_search_words]
    if not chunk_words:
        return None
    
    search_text = ' '.join(chunk_words).lower().strip()
    
    # Find the position in the existing transcript_text
    search_pos = transcript_text.lower().find(search_text)
    if search_pos == -1:
        # If exact match fails, try with fewer words
        for i in range(min(5, len(chunk_words)), 0, -1):
            shorter_search = ' '.join(chunk_words[:i]).lower().strip()
            search_pos = transcript_text.lower().find(shorter_search)
            if search_pos != -1:
                break
    
    if search_pos == -1:
        logging.error(f"Could not find chunk text in transcript")
        logging.error(f"Chunk start: '{chunk_text[:100]}'")
        logging.error(f"Search text tried: '{search_text}'")
        return None
    
    # Map the position back to the corresponding timestamp in transcript_json
    current_pos = 0
    for i, item in enumerate(transcript_json):
        segment_text = item.get('text', '')
        segment_end = current_pos + len(segment_text)
        
        # Check if our search position falls within this segment
        if current_pos <= search_pos < segment_end:
            return item.get('start', 0.0)
        
        # Move to next position (segment text + space separator)
        current_pos = segment_end
        # Add space separator if not the last segment
        if i < len(transcript_json) - 1:
            current_pos += 1
    
    return None

def split_documents_into_chunks(documents, chunk_size, chunk_overlap):
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    for doc in documents:
        video_id = doc.metadata.get('video_id')
        # Get the transcript_json before splitting
        transcript_json = doc.metadata.get('transcript_json')
        
        # Use split_documents to preserve metadata
        chunks_with_metadata = splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks_with_metadata):
            # Clean the chunk content
            chunk.page_content = chunk.page_content.strip()
            if not chunk.page_content:
                continue

            # Find start time using text matching - MUST succeed
            start_time = find_start_time_by_text(chunk.page_content, transcript_json, doc.page_content)
            
            if start_time is None:
                logging.error(f"CRITICAL ERROR: Failed to find start time for chunk {i} in video {video_id}")
                logging.error(f"Chunk content preview: '{chunk.page_content[:100]}...'")
                raise RuntimeError(f"Could not find start time for chunk {i} in video {video_id}. This indicates a fundamental problem with the text matching logic.")
            
            # Update chunk metadata
            chunk.metadata['start_time'] = start_time
            chunk.metadata['chunk_index'] = i
            # We no longer need the original transcript JSON
            chunk.metadata.pop('transcript_json', None)
        
        all_chunks.extend(chunks_with_metadata)

    # Sort by start_time - all should be valid now
    all_chunks.sort(key=lambda x: x.metadata['start_time'])

    logging.debug(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def store_chunks_locally(chunks, chunks_file):
    logging.debug(f"Storing {len(chunks)} chunks locally to {chunks_file}")
    with open(chunks_file, "w") as f:
        json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in chunks], f)
    logging.debug(f"Successfully stored chunks in {chunks_file}")

# --- Embeddings ---

def create_embeddings(chunks, batch_size=100):
    client = OpenAI()
    embeddings = []
    logging.debug(f"Creating embeddings for {len(chunks)} chunks...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Creating embeddings"):
        batch = chunks[i:i+batch_size]
        texts = [chunk.page_content for chunk in batch]
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    logging.debug("Embeddings created successfully.")
    return np.array(embeddings)

def store_embeddings_locally(embeddings, embeddings_file):
    logging.debug(f"Storing embeddings locally to {embeddings_file}")
    np.save(embeddings_file, embeddings)
    logging.debug(f"Successfully stored embeddings in {embeddings_file}")

# --- Comprehensive Test Functions ---
def test_data_integrity(chunks, video_id):
    """Test that all chunks have complete and valid metadata. FAIL HARD on any issues."""
    required_fields = ['start_time', 'chunk_index', 'video_id', 'live_number', 'published_at', 'title']
    
    for i, chunk in enumerate(chunks):
        # Check required metadata fields
        for field in required_fields:
            if field not in chunk.metadata:
                error_msg = f"INTEGRITY TEST FAILED: Chunk {i} missing required field '{field}'. Video: {video_id}"
                logging.error(error_msg)
                raise RuntimeError(error_msg)
        
        # Check start_time validity
        start_time = chunk.metadata.get('start_time')
        if start_time is None:
            error_msg = f"INTEGRITY TEST FAILED: Chunk {i} has None start_time. Video: {video_id}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        elif start_time < 0:
            error_msg = f"INTEGRITY TEST FAILED: Chunk {i} has negative start_time: {start_time}. Video: {video_id}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Check content validity
        if not chunk.page_content or not chunk.page_content.strip():
            error_msg = f"INTEGRITY TEST FAILED: Chunk {i} has empty content. Video: {video_id}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    
    logging.debug(f"Data integrity test PASSED for {len(chunks)} chunks in video {video_id}")
    return True

def test_token_counts(chunks, video_id, chunk_size, chunk_overlap):
    """Test that token counts are within reasonable limits. Allow some wiggle room for text boundary preservation."""
    
    # Allow up to 10% over the target size, or minimum 50 tokens
    max_allowed = chunk_size + max(int(chunk_size * 0.1), 50)
    
    for i, chunk in enumerate(chunks):
        token_count = num_tokens_from_string(chunk.page_content)
        
        # Check if chunk exceeds reasonable token size
        if token_count > max_allowed:
            error_msg = f"TOKEN TEST FAILED: Chunk {i} has {token_count} tokens, exceeds reasonable limit of {max_allowed} (target: {chunk_size}). Video: {video_id}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log if chunk is over target but within acceptable range
        if token_count > chunk_size:
            logging.debug(f"  Chunk {i} slightly over target: {token_count} tokens (target: {chunk_size}, acceptable)")
        
        # Log token count for first few chunks
        if i < 3:
            logging.debug(f"  Chunk {i} token count: {token_count}")
    
    logging.debug(f"Token count test PASSED for {len(chunks)} chunks in video {video_id} (max allowed: {max_allowed})")
    return True

def test_text_continuity(chunks, original_text, video_id, chunk_overlap):
    """Test that chunks properly represent the original text with correct overlap."""
    test_passed = True
    
    if not chunks:
        return test_passed
    
    # Check that first chunk starts at the beginning of the original text
    first_chunk_start = chunks[0].page_content[:50].strip()
    original_start = original_text[:50].strip()
    
    if not original_start.lower().startswith(first_chunk_start.lower()[:30]):
        logging.debug(f"CONTINUITY WARNING: First chunk doesn't start at beginning of original text. Video: {video_id}")
        logging.debug(f"Original start: '{original_start}'")
        logging.debug(f"First chunk start: '{first_chunk_start}'")
    
    # Check for overlap between consecutive chunks
    total_chunks = len(chunks)
    failed_overlaps = 0
    
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].page_content.strip()
        next_chunk = chunks[i + 1].page_content.strip()
        
        # Try different overlap detection strategies
        overlap_found = False
        
        # Strategy 1: Check if end of current chunk appears in start of next chunk
        # Use larger sections for 200 token overlap (~800-1000 characters)
        for check_length in [1000, 800, 500, 300, 200]:  # Much larger lengths
            if check_length >= len(current_chunk) or check_length >= len(next_chunk):
                continue
            
            current_end = current_chunk[-check_length:].strip()
            next_start = next_chunk[:check_length].strip()
            
            # Normalize whitespace for comparison
            current_end_norm = ' '.join(current_end.split())
            next_start_norm = ' '.join(next_start.split())
            
            # Check if there's overlap (at least 50 characters for meaningful overlap)
            for j in range(min(len(current_end_norm), len(next_start_norm)), 49, -1):
                if current_end_norm[-j:].lower() == next_start_norm[:j].lower():
                    overlap_found = True
                    if i < 2:  # Log for first few chunks
                        logging.debug(f"  Overlap found between chunks {i} and {i+1}: '{current_end_norm[-j:]}' ({j} chars)")
                    break
            
            if overlap_found:
                break
        
        # Strategy 2: Check if start of next chunk appears anywhere in end of current chunk
        if not overlap_found:
            # Try finding first few words of next chunk in current chunk
            next_words = next_chunk.split()[:10]  # More words
            if next_words:
                search_text = ' '.join(next_words).lower()
                if search_text in current_chunk.lower():
                    overlap_found = True
                    if i < 2:
                        logging.debug(f"  Word-based overlap found between chunks {i} and {i+1}")
        
        if not overlap_found:
            failed_overlaps += 1
            if failed_overlaps <= 3:  # Only log first few failures to avoid spam
                logging.debug(f"CONTINUITY WARNING: No clear overlap found between chunks {i} and {i+1}. Video: {video_id}")
                # Show much more text for debugging (enough to see 200 token overlap)
                current_end = current_chunk[-800:] if len(current_chunk) > 800 else current_chunk
                next_start = next_chunk[:800] if len(next_chunk) > 800 else next_chunk
                logging.debug(f"  Chunk {i} end (last 800 chars): '...{current_end}'")
                logging.debug(f"  Chunk {i+1} start (first 800 chars): '{next_start}...'")
    
    if failed_overlaps > 0:
        logging.debug(f"CONTINUITY: {failed_overlaps}/{total_chunks-1} chunk pairs have no detectable overlap")
        test_passed = False
    
    return test_passed

def test_start_time_accuracy(chunks, video_id):
    """Test that start times are reasonable and generally increasing."""
    test_passed = True
    
    if len(chunks) < 2:
        return test_passed
    
    start_times = [chunk.metadata.get('start_time') for chunk in chunks if chunk.metadata.get('start_time') is not None]
    
    if not start_times:
        logging.error(f"START_TIME TEST FAILED: No valid start times found. Video: {video_id}")
        return False
    
    # Check that start times are generally increasing (allow some flexibility for overlaps)
    decreases = 0
    for i in range(len(start_times) - 1):
        if start_times[i+1] < start_times[i]:
            decreases += 1
    
    decrease_percentage = decreases / (len(start_times) - 1) if len(start_times) > 1 else 0
    
    if decrease_percentage > 0.2:  # Allow up to 20% decreases due to overlaps
        logging.warning(f"START_TIME WARNING: {decrease_percentage:.1%} of start times decrease. Video: {video_id}")
        test_passed = False
    
    # Log some start time info
    logging.debug(f"  Start times: {start_times[0]:.1f}s -> {start_times[-1]:.1f}s ({len(start_times)} chunks)")
    
    return test_passed

def test_chunking_logic(documents_to_process, chunk_size, chunk_overlap):
    """
    Comprehensive test suite for chunking logic.
    """
    logging.info("=== RUNNING COMPREHENSIVE CHUNKING TESTS ===")
    all_tests_passed = True
    
    # Test a sample of documents
    sample_docs = documents_to_process if len(documents_to_process) < 3 else documents_to_process[:3]
    
    for doc in sample_docs:
        video_id = doc.metadata.get("video_id")
        logging.info(f"\n--- Testing document: {video_id} ---")
        
        # Split the document
        chunks = split_documents_into_chunks([doc], chunk_size, chunk_overlap)
        
        if not chunks:
            logging.warning(f"No chunks created for video_id: {video_id}. Skipping tests.")
            continue
        
        logging.info(f"Created {len(chunks)} chunks for testing")
        
        # Run all test suites
        tests = [
            ("Data Integrity", test_data_integrity(chunks, video_id)),
            ("Token Counts", test_token_counts(chunks, video_id, chunk_size, chunk_overlap)),
            ("Text Continuity", test_text_continuity(chunks, doc.page_content, video_id, chunk_overlap)),
            ("Start Time Accuracy", test_start_time_accuracy(chunks, video_id))
        ]
        
        # Report results for this document
        for test_name, test_result in tests:
            if test_result:
                logging.info(f"  ✓ {test_name} test PASSED")
            else:
                logging.error(f"  ✗ {test_name} test FAILED")
                all_tests_passed = False
        
        # Log sample chunk info
        if chunks:
            chunk = chunks[0]
            log_meta = chunk.metadata.copy()
            log_meta.pop('transcript_json', None)
            logging.info(f"  Sample chunk metadata: {log_meta}")
            logging.info(f"  Sample chunk preview: {chunk.page_content[:100]}...")
    
    if all_tests_passed:
        logging.info("\n=== ALL COMPREHENSIVE TESTS PASSED ===")
    else:
        logging.error("\n=== SOME COMPREHENSIVE TESTS FAILED ===")
    
    return all_tests_passed

# --- Main ---

def main():
    chunk_size = 1000
    chunk_overlap = 200
    live_start_number = 750

    setup_logging(chunk_size, chunk_overlap)
    print(f"✓ Starting chunking process (chunk_size: {chunk_size}, overlap: {chunk_overlap})")

    # --- Setup ---
    folder_path = f'data/size-{chunk_size}-ol-{chunk_overlap}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    input_file = 'data/video-data-with-transcripts.json'
    
    logging.info(f"Data folder: {folder_path}")
    logging.info(f"Input file: {input_file}")
    
    # --- Process ---
    print("📂 Loading documents...")
    documents = load_documents(input_file)
    original_doc_count = len(documents)
    documents_to_process = [doc for doc in documents if doc.metadata.get('live_number') and doc.metadata.get('live_number') >= live_start_number]
    print(f"✓ Loaded {original_doc_count} documents, filtered to {len(documents_to_process)} (live_number >= {live_start_number})")
    logging.info(f"Filtered documents from {original_doc_count} to {len(documents_to_process)} (live_number >= {live_start_number}).")

    print("🧪 Running comprehensive tests...")
    if not test_chunking_logic(documents_to_process, chunk_size, chunk_overlap):
        logging.error("Aborting due to test failure.")
        return
    print("✓ All tests passed!")

    print(f"🚀 Processing {len(documents_to_process)} documents...")
    for i, doc in enumerate(tqdm(documents_to_process, desc="Processing documents"), 1):
        video_id = doc.metadata.get("video_id")
        print(f"  📹 Processing video {i}/{len(documents_to_process)}: {video_id}")
        logging.info(f"Processing document: {video_id}")
        
        chunks = split_documents_into_chunks([doc], chunk_size, chunk_overlap)
        
        if not chunks:
            logging.error(f"No chunks created for video_id: {video_id}. This indicates a fundamental problem.")
            raise RuntimeError(f"Failed to create chunks for video_id: {video_id}")
        
        # Validate that ALL chunks have valid start times - fail immediately if not
        for j, chunk in enumerate(chunks):
            start_time = chunk.metadata.get('start_time')
            if start_time is None:
                logging.error(f"Chunk {j} for video_id {video_id} has None start_time. This is unacceptable.")
                raise RuntimeError(f"Chunk {j} for video_id {video_id} missing start_time")
            if start_time < 0:
                logging.error(f"Chunk {j} for video_id {video_id} has invalid start_time: {start_time}")
                raise RuntimeError(f"Chunk {j} for video_id {video_id} has invalid start_time: {start_time}")
        
        print(f"    ✓ Created {len(chunks)} chunks")
        logging.info(f"All {len(chunks)} chunks have valid start times")
        
        print(f"    🔮 Creating embeddings...")
        embeddings = create_embeddings(chunks)
        
        print(f"    💾 Storing chunks and embeddings locally...")
        chunks_file = os.path.join(folder_path, 'chunks.json')
        embeddings_file = os.path.join(folder_path, 'embeddings.npy')
        store_chunks_locally(chunks, chunks_file)
        store_embeddings_locally(embeddings, embeddings_file)
        
        print(f"    ✅ Completed video {video_id}")
        logging.info(f"Successfully processed and stored {len(chunks)} chunks for video_id: {video_id}")

    print("🎉 Processing complete!")
    logging.info("Processing complete.")

if __name__ == "__main__":
    main() 