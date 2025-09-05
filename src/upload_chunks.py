import os
import json
import time
import random
from pinecone import Pinecone
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

def upload_chunks_to_pinecone():
    """
    Loads chunks from a JSON file and uploads them to a Pinecone index
    where embeddings are generated automatically.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_HOST")
    pinecone_index_name = os.getenv("PINECONE_INDEX")
    chunks_file = 'data/size-1000-ol-200/chunks.json'

    if not all([pinecone_api_key, pinecone_host, pinecone_index_name]):
        print("Error: Pinecone environment variables (PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX) must be set.")
        return

    print("✓ Pinecone environment variables loaded.")
    print(f"  - Index: {pinecone_index_name}")
    print(f"  - Host: {pinecone_host}")
    print(f"  - Chunks file: {chunks_file}")

    # 1. Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # 2. Connect to the index
    print(f"\n🔗 Connecting to Pinecone index '{pinecone_index_name}'...")
    try:
        index = pc.Index(host=pinecone_host)
        stats = index.describe_index_stats()
        print(f"✓ Successfully connected. Index has {stats['total_vector_count']} vectors.")
    except Exception as e:
        print(f"Error connecting to Pinecone index: {e}")
        print("Please ensure the index exists and the host is correct. This script will not create a new index.")
        return

    # 3. Load chunks from local file
    print(f"📂 Loading chunks from '{chunks_file}'...")
    try:
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        print(f"✓ Loaded {len(chunks)} chunks.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading chunks file: {e}")
        return

    # 4. Prepare and upsert data in batches
    batch_size = 96  # As recommended by Pinecone for text-based upserts
    print(f"🚀 Upserting {len(chunks)} chunks to Pinecone in batches of {batch_size}...")

    vectors_to_upsert = []
    for chunk in chunks:
        unique_id = f'{chunk["metadata"]["video_id"]}-{chunk["metadata"]["chunk_index"]}'
        
        data = {
            "_id": unique_id,
            "text": chunk["page_content"],
            "video_id": chunk["metadata"].get("video_id"),
            "live_number": chunk["metadata"].get("live_number"),
            "published_at": chunk["metadata"].get("published_at"),
            "title": chunk["metadata"].get("title"),
            "start_time": chunk["metadata"].get("start_time"),
            "chunk_index": chunk["metadata"].get("chunk_index")
        }
        
        vectors_to_upsert.append(data)

    for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Uploading batches"):
        batch = vectors_to_upsert[i:i + batch_size]
        
        # Implement retry logic with exponential backoff
        max_retries = 6
        initial_delay = 5  # seconds
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                index.upsert_records(namespace="default", records=batch)
                # If success, break out of the retry loop for this batch
                break
            except Exception as e:
                error_message = str(e).lower()
                # Check for rate limiting error message
                if '429' in error_message or 'tokens per minute' in error_message:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit. Retrying batch in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        print(f"Max retries reached for batch. Skipping. Error: {e}")
                        # This break is for the retry loop, we will continue with the next batch.
                        break
                else:
                    # Handle other types of errors
                    print(f"An unexpected error occurred: {e}")
                    # Decide if you want to break or continue on other errors
                    break
    
    # 5. Final verification
    try:
        final_stats = index.describe_index_stats()
        print("\n🎉 Upload complete!")
        print(f"✓ Index now has {final_stats['total_vector_count']} vectors.")
    except Exception as e:
        print(f"Could not retrieve final index stats: {e}")


if __name__ == "__main__":
    upload_chunks_to_pinecone()
