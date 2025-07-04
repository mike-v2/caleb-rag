import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter, TextFormatter
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

def add_transcripts_to_video_data(video_data):
    """Fetch and add transcripts to video data."""
    json_formatter = JSONFormatter()
    text_formatter = TextFormatter()
    
    videos_with_transcripts = []

    for i, video in enumerate(video_data):
        video_id = video['video_id']
        try:
            print(f"({i+1}/{len(video_data)}) Getting transcripts for video: {video_id} (Live #{video.get('live_number', 'N/A')})")
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            json_formatted = json_formatter.format_transcript(transcript)
            video['transcript_json'] = json.loads(json_formatted)
            
            text = text_formatter.format_transcript(transcript).replace('\n', ' ')
            video['transcript_text'] = text
            
            videos_with_transcripts.append(video)
        except Exception as e:
            print(f"Failed to get transcript for video {video_id}: {str(e)}")
            # We don't add the video if transcript fetching fails
        
    return videos_with_transcripts

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

    new_videos_with_transcripts = add_transcripts_to_video_data(videos_to_process)

    if new_videos_with_transcripts:
        print(f"Successfully fetched transcripts for {len(new_videos_with_transcripts)} videos.")
        
        # Combine and save
        combined_data = videos_with_transcripts + new_videos_with_transcripts
        combined_data.sort(key=lambda x: x.get('live_number', 0), reverse=True)
        save_json_file(combined_data, transcripts_path)
        
        print(f"Updated data with transcripts written to {transcripts_path}")
        print(f"Total videos with transcripts: {len(combined_data)}")
    else:
        print("No new transcripts were fetched.")

if __name__ == '__main__':
    main() 