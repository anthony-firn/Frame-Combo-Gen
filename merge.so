import yt_dlp as ydl
import pyscenedetect
from pyscenedetect import VideoManager, SceneManager, detection
import os
import cv2
import json
from tqdm import tqdm
import numpy as np
from pyscenedetect.detectors import ContentDetector
from pyscenedetect.detectors import ContentDetector
from pyscenedetect.video_splitter import split_video_ffmpeg

# Constants
FPS = 30
THRESHOLD = 50  # Example threshold for static scene detection

# Download videos using yt-dlp for multiple search terms
def download_videos(search_terms, num_results_per_term):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',  # Ensure videos are downloaded in MP4 format
        'noplaylist': True,
        'outtmpl': './videos/%(title)s.%(ext)s',
        'quiet': False,
        'postprocessors': [],
    }

    with ydl.YoutubeDL(ydl_opts) as ydl_instance:
        for term in search_terms:
            ydl_instance.download([f"ytsearch{num_results_per_term}:{term}"])

# Filter videos based on frame rate
def filter_videos_by_fps(directory, target_fps=FPS):
    valid_videos = []
    for video in os.listdir(directory):
        video_path = os.path.join(directory, video)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if fps == target_fps:
            valid_videos.append(video_path)
    return valid_videos

# Extract metadata from video and scene
def extract_metadata(video_path, scene_path, x, y):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    scene_cap = cv2.VideoCapture(scene_path)
    scene_duration = int(scene_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    scene_cap.release()

    return {
        'source_url': video_path,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'duration': duration,
        'codec': codec,
        'x_value': x,
        'y_value': y,
        'scene_duration': scene_duration
    }

# Extract and split scenes using pyscenedetect
def extract_and_split_scenes(video_path, output_dir):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    # Split video into scenes and save them
    scene_list = scene_manager.get_scene_list(video_path)
    split_video_ffmpeg(video_manager, scene_list, output_dir, video_name="scene_$SCENE_NUMBER.mp4")
    
    return [os.path.join(output_dir, f"scene_{i}.mp4") for i in range(len(scene_list))]

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
def extract_frames(scene_path, x, y):
    cap = cv2.VideoCapture(scene_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    # Adjusted logic to handle different 'x' and 'y' values
    for i in range(0, total_frames, x):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)
        for j in range(1, y):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i + j)
            ret, frame = cap.read()
            if ret:
                extracted_frames.append(frame)

    cap.release()
    return extracted_frames

# Combine frames into a single image
def combine_frames(frames):
    return cv2.hconcat(frames)

# Organize combined images into dataset folders
def organize_images(image, x, y):
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
    search_terms = input("Enter the search terms separated by commas: ").split(',')
    num_results = int(input("Enter the number of results to fetch per term: "))

    download_videos(search_terms, num_results)

    valid_videos = filter_videos_by_fps('./videos')

    metadata_list = []

    for video in tqdm(valid_videos, desc="Processing videos"):
        video_path = os.path.join('./videos', video)
        
        scenes = extract_and_split_scenes(video_path, './scenes')
        for scene in scenes:
            process_scene(scene, x=2, y=2)  # Example values

            # Extract metadata for each scene
            metadata = extract_metadata(video_path, scene, x=2, y=2)
            metadata_list.append(metadata)

    # Save metadata to a JSON file
    with open('metadata.json', 'w') as f:
        json.dump(metadata_list, f)

if __name__ == "__main__":
    main()