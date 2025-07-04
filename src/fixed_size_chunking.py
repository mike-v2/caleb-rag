import os
import json
import time
import argparse
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import JSONLoader
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

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
    print(f'Loaded {len(documents)} documents.')
    return documents

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def find_start_time(chunk_text, transcript_json):
    chunk_words = chunk_text.split()
    if not transcript_json:
        return None
    transcript_words = [word for item in transcript_json for word in item['text'].split()]
    transcript_start_times = [item['start'] for item in transcript_json for _ in item['text'].split()]

    for i, word in enumerate(transcript_words):
        if word == chunk_words[0]:
            if transcript_words[i:i+5] == chunk_words[:5]:
                return transcript_start_times[i]
    return None

def split_documents_into_chunks(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=num_tokens_from_string,
        add_start_index=True
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc.page_content)
        current_start_index = 0
        for i, chunk in enumerate(doc_chunks):
            start_time = find_start_time(chunk, doc.metadata.get('transcript_json'))
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    'video_id': doc.metadata.get('video_id'),
                    'live_number': doc.metadata.get('live_number'),
                    'published_at': doc.metadata.get('published_at'),
                    'title': doc.metadata.get('title'),
                    'chunk_index': i,
                    'start_time': start_time,
                    'start_index': current_start_index
                }
            ))
            current_start_index += len(chunk)
            if i < len(doc_chunks) - 1:
                current_start_index -= chunk_overlap

    chunks.sort(key=lambda x: x.metadata['start_time'] if x.metadata.get('start_time') is not None else float('inf'))
    
    print(f"Total chunks: {len(chunks)}")
    return chunks

def store_chunks_locally(chunks, chunks_file):
    with open(chunks_file, "w") as f:
        json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in chunks], f)

# --- Embeddings ---

def create_embeddings(chunks, batch_size=100):
    client = OpenAI()
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        texts = [chunk.page_content for chunk in batch]
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def store_embeddings_locally(embeddings, embeddings_file):
    np.save(embeddings_file, embeddings)

# --- Pinecone ---

def get_pinecone_index(index_name):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        print(f'Creating index: {index_name}')
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    return pc.Index(index_name)

def upsert_to_pinecone(index, chunks, embeddings):
    def prepare_data(chunks, embeddings):
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            yield (
                str(i),
                embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                {
                    "text": chunk.page_content,
                    "video_id": chunk.metadata.get("video_id"),
                    "live_number": chunk.metadata.get("live_number"),
                    "published_at": chunk.metadata.get("published_at"),
                    "title": chunk.metadata.get("title"),
                    "start_time": chunk.metadata.get("start_time"),
                    "chunk_index": chunk.metadata.get("chunk_index")
                }
            )

    batch_size = 96
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_data = list(prepare_data(chunks[i:i+batch_size], embeddings[i:i+batch_size]))
        index.upsert(vectors=batch_data)
    print(index.describe_index_stats())

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Chunk transcripts, create embeddings, and upsert to Pinecone.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="The size of each text chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="The overlap between consecutive chunks.")
    args = parser.parse_args()

    # --- Setup ---
    base_folder = '../data/fixed_size_chunks'
    folder_path = os.path.join(base_folder, f'size-{args.chunk_size}-ol-{args.chunk_overlap}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    chunks_file = os.path.join(folder_path, 'chunks.json')
    embeddings_file = os.path.join(folder_path, 'embeddings.npy')
    input_file = '../data/video-data-with-transcripts.json'
    
    # --- Process ---
    documents = load_documents(input_file)
    chunks = split_documents_into_chunks(documents, args.chunk_size, args.chunk_overlap)
    store_chunks_locally(chunks, chunks_file)
    
    embeddings = create_embeddings(chunks)
    store_embeddings_locally(embeddings, embeddings_file)
    
    index_name = f"caleb-rag-fixed-{args.chunk_size}-{args.chunk_overlap}"
    index = get_pinecone_index(index_name)
    upsert_to_pinecone(index, chunks, embeddings)

    print("Processing complete.")

if __name__ == "__main__":
    main() 