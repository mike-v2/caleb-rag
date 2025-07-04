import os
import re
import json
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter, TextFormatter
from dotenv import load_dotenv

load_dotenv()

def load_existing_video_data(file_path):
    """Load existing video data if it exists"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        print(f"Loaded {len(existing_data)} existing videos from {file_path}")
        return existing_data
    except FileNotFoundError:
        print(f"No existing data found at {file_path}, starting fresh")
        return []

def get_video_data(api_key, channel_id, existing_videos=None):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    if existing_videos is None:
        existing_videos = []
    
    # Create a set of existing video IDs for quick lookup
    existing_video_ids = {video['video_id'] for video in existing_videos}
    existing_live_numbers = {video['live_number'] for video in existing_videos}
    
    print(f"Existing videos: {len(existing_videos)}")
    print(f"Highest existing live number: {max(existing_live_numbers) if existing_live_numbers else 0}")
    
    video_data = []
    next_page_token = None

    channels_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    while True:
        print(f'Retrieving page: {next_page_token}')
        playlist_items_response = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in playlist_items_response['items']:
            title = item['snippet']['title']
            if title.startswith('Live'):
                match = re.search(r'Live\s*#?\s*(\d+)', title)
                if match:
                    live_number = int(match.group(1))
                    video_id = item['contentDetails']['videoId']
                    published_at = item['snippet']['publishedAt']
                    
                    # Skip if we already have this video
                    if video_id in existing_video_ids:
                        continue
                    
                    # Skip if we already have this live number (in case of different video IDs)
                    if live_number in existing_live_numbers:
                        print(f"Skipping Live #{live_number} - already exists")
                        continue
                    
                    video_data.append({
                        'video_id': video_id,
                        'title': title,
                        'live_number': live_number,
                        'published_at': published_at
                    })
                    print(f"Found new video: Live #{live_number} - {title}")

        next_page_token = playlist_items_response.get('nextPageToken')
        
        if not next_page_token:
            break
    
    print(f'Found {len(video_data)} new videos')
    return video_data

def main():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in environment variables.")
        
    channel_id = 'UCA0N4UVTnqiFB7D3oS_xdWg'
    
    video_data_path = '../data/video-data.json'
    
    # Load existing video data
    existing_videos = load_existing_video_data(video_data_path)
    
    # Get new video data
    new_videos = get_video_data(api_key, channel_id, existing_videos)
    
    if new_videos:
        # Combine existing and new videos
        all_videos = existing_videos + new_videos
        
        # Sort by live number (descending)
        all_videos.sort(key=lambda x: x['live_number'], reverse=True)
        
        # Save combined data
        with open(video_data_path, 'w', encoding='utf-8') as f:
            json.dump(all_videos, f, ensure_ascii=False, indent=4)
        print(f"Combined video data saved to {video_data_path} ({len(all_videos)} total videos)")

        print("\nNext, run the get_transcripts.py script to fetch transcripts for the new videos.")
    else:
        print("No new videos found!")

if __name__ == '__main__':
    main() 