import yt_dlp as ydl
import pyscenedetect
from pyscenedetect import VideoManager, SceneManager, detection
import os
import cv2
import json
from tqdm import tqdm
import numpy as np

# Constants
FPS = 30
THRESHOLD = 50  # Example threshold for static scene detection

# Download videos using yt-dlp
def download_videos(search_term, num_results):
    ydl_opts = {
        'format': 'bestvideo',
        'noplaylist': True,
        'outtmpl': './videos/%(title)s.%(ext)s',
        'quiet': False,
        'postprocessors': [],
        'default_search': 'ytsearch' + str(num_results)
    }

    with ydl.YoutubeDL(ydl_opts) as ydl_instance:
        ydl_instance.download([search_term])

# Extract scenes using pyscenedetect
def extract_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(detection.ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()

# Check if scene is static
def is_static_scene(scene):
    total_diff = 0
    for i in range(1, len(scene)):
        diff = cv2.absdiff(scene[i-1], scene[i])
        total_diff += np.sum(diff)
    avg_diff = total_diff / len(scene)
    return avg_diff < THRESHOLD

# Process each scene
def process_scene(scene, x, y):
    # Extract frames based on x and y
    frames = extract_frames(scene, x, y)
    
    # Check if scene is static
    if is_static_scene(frames):
        return
    
    # Combine frames into single images
    combined_image = combine_frames(frames)
    
    # Organize combined images into dataset folders
    organize_images(combined_image, x, y)

# Extract frames from a scene based on x and y values
def extract_frames(scene, x, y):
    # Implement frame extraction logic based on x and y values
    # Placeholder logic
    extracted_frames = []
    for i in range(0, len(scene), x):
        extracted_frames.append(scene[i:i+y])
    return extracted_frames

# Combine frames into a single image
def combine_frames(frames):
    # Implement logic to combine frames into a single image
    # Placeholder logic
    return np.hstack(frames)

# Organize combined images into dataset folders
def organize_images(image, x, y):
    # Implement logic to organize images into dataset folders
    # Placeholder logic
    folder_name = f"x_{x}_y_{y}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    cv2.imwrite(os.path.join(folder_name, f"combined_{x}_{y}.jpg"), image)

# Extract metadata from video
def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    return {
        'fps': fps,
        'resolution': f"{width}x{height}",
        'duration': duration,
        'codec': codec
    }

# Main function
def main():
    search_term = input("Enter the search term: ")
    num_results = int(input("Enter the number of results to fetch: "))

    download_videos(search_term, num_results)

    metadata_list = []

    for video in tqdm(os.listdir('./videos'), desc="Processing videos"):
        video_path = os.path.join('./videos', video)
        
        # Extract metadata
        metadata = extract_metadata(video_path)
        if metadata['fps'] != FPS:
            continue

        scenes = extract_scenes(video_path)
        for scene in scenes:
            process_scene(scene, x=2, y=2)  # Example values

        metadata_list.append(metadata)

    # Save metadata to a JSON file
    with open('metadata.json', 'w') as f:
        json.dump(metadata_list, f)

if __name__ == "__main__":
    main()