import os
import json
import logging

from .config import (
    INPUT_JSON_PATH, OUTPUT_JSON_PATH,
    LOGS_DIR, LOG_FILE_PATH, LOG_LEVEL, LOG_FORMAT
)
from .chunking import process_video

def setup_logging():
    """Configures the logging for the application."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT
    )

def main():
    """Main function to run the chunking process."""
    setup_logging()
    
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            videos = json.load(f)
        logging.info(f"Loaded {len(videos)} videos from {INPUT_JSON_PATH}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_JSON_PATH}")
        print(f"Error: Input file not found at '{INPUT_JSON_PATH}'.")
        return
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {INPUT_JSON_PATH}")
        print(f"Error: Could not decode JSON from '{INPUT_JSON_PATH}'.")
        return

    all_chunks = []
    for video in videos:
        video_id = video.get('video_id')
        if not video_id:
            logging.warning("Skipping video with no video_id.")
            continue

        monologue_chunks, qa_chunks = process_video(video)

        for chunk in monologue_chunks:
            all_chunks.append({
                "video_id": video_id,
                "chunk_type": "monologue",
                **chunk
            })

        for chunk in qa_chunks:
            all_chunks.append({
                "video_id": video_id,
                "chunk_type": "q_a",
                **chunk
            })

    # Save all processed chunks
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved {len(all_chunks)} chunks to {OUTPUT_JSON_PATH}")
        print(f"\nProcessing complete. Saved {len(all_chunks)} chunks to {OUTPUT_JSON_PATH}")
    except IOError as e:
        logging.error(f"Failed to write output file: {e}")
        print(f"Error: Could not write to output file at '{OUTPUT_JSON_PATH}'.")


if __name__ == "__main__":
    main() 