import json
import random
import time
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()

def load_json_file(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    """Main function to get missing transcripts."""
    video_data_path = 'data/video-data.json'
    transcripts_path = 'data/video-data-with-transcripts.json'

    all_videos = load_json_file(video_data_path)
    videos_with_transcripts = load_json_file(transcripts_path)

    if not all_videos:
        print("No video data found. Please run the script to get video data first.")
        return

    existing_transcript_ids = {v['video_id'] for v in videos_with_transcripts}
    
    videos_to_process = [
        video for video in all_videos 
        if video['video_id'] not in existing_transcript_ids
    ]

    if not videos_to_process:
        print("All videos already have transcripts. No new transcripts to fetch.")
        return

    print(f"Found {len(videos_to_process)} videos that need transcripts.")

    successful_fetches = 0
    total_to_process = len(videos_to_process)

    for i, video in enumerate(videos_to_process):
        delay = random.uniform(30, 60)
        print(f"Waiting for {delay:.2f} seconds before next request...")
        time.sleep(delay)

        video_id = video['video_id']
        try:
            print(f"({i+1}/{total_to_process}) Getting transcripts for video: {video_id} (Live #{video.get('live_number', 'N/A')})")
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            video['transcript_json'] = transcript
            text = ' '.join([item.get('text', '') for item in transcript])
            video['transcript_text'] = text

            videos_with_transcripts.append(video)
            successful_fetches += 1
            
            # Sort and save after each successful fetch
            videos_with_transcripts.sort(key=lambda x: x.get('live_number', 0), reverse=True)
            save_json_file(videos_with_transcripts, transcripts_path)
            
            print(f"Successfully fetched and saved transcript for video {video_id}.")
            print(f"Total videos with transcripts: {len(videos_with_transcripts)}")

        except Exception as e:
            print(f"Failed to get transcript for video {video_id}: {str(e)}")

    if successful_fetches > 0:
        print(f"\nFinished processing. Successfully fetched transcripts for {successful_fetches} new videos.")
        print(f"Updated data with transcripts is in {transcripts_path}")
        print(f"Total videos with transcripts: {len(videos_with_transcripts)}")
    else:
        print("\nFinished processing. No new transcripts were fetched.")

if __name__ == '__main__':
    main() 