import os
import re
import json
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter, TextFormatter
from dotenv import load_dotenv

load_dotenv()

def get_video_data(api_key, channel_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
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
                    live_number = match.group(1)
                    video_id = item['contentDetails']['videoId']
                    published_at = item['snippet']['publishedAt']
                    video_data.append({
                        'video_id': video_id,
                        'title': title,
                        'live_number': int(live_number),
                        'published_at': published_at
                    })

        next_page_token = playlist_items_response.get('nextPageToken')
        
        if not next_page_token:
            break
    
    print(f'Found {len(video_data)} videos')
    return video_data

def add_transcripts_to_video_data(video_data):
    json_formatter = JSONFormatter()
    text_formatter = TextFormatter()

    for video in video_data:
        video_id = video['video_id']
        try:
            print(f"Getting transcripts for video: {video_id}")
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            json_formatted = json_formatter.format_transcript(transcript)
            video['transcript_json'] = json.loads(json_formatted)
            
            text = text_formatter.format_transcript(transcript).replace('\n', ' ')
            video['transcript_text'] = text
        except Exception as e:
            print(f"Failed to get transcript for video {video_id}: {str(e)}")
            video['transcript_json'] = None
            video['transcript_text'] = None
    return video_data

def main():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in environment variables.")
        
    channel_id = 'UCA0N4UVTnqiFB7D3oS_xdWg'
    
    # Get video data
    videos = get_video_data(api_key, channel_id)
    with open('../data/video-data.json', 'w', encoding='utf-8') as f:
        json.dump(videos, f, ensure_ascii=False, indent=4)
    print(f"Video data saved to data/video-data.json")

    # Add transcripts
    videos_with_transcripts = add_transcripts_to_video_data(videos)
    with open('../data/video-data-with-transcripts.json', 'w', encoding='utf-8') as f:
        json.dump(videos_with_transcripts, f, ensure_ascii=False, indent=4)
    print(f"Updated data with transcripts written to data/video-data-with-transcripts.json")

if __name__ == '__main__':
    main() 