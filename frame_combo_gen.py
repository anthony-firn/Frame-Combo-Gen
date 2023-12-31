# Import necessary libraries and modules
import yt_dlp as ydl
import pyscenedetect
from pyscenedetect import VideoManager, SceneManager, detection
import os
import cv2
import json
from tqdm import tqdm
import numpy as np
from pyscenedetect.detectors import ContentDetector
from pyscenedetect.video_splitter import split_video_ffmpeg
import logging

# Set up logging to capture both INFO and ERROR level logs
logging.basicConfig(filename='dataset_generator.log', level=logging.INFO)

# Define constants
FPS = 30  # Target frames per second for videos
THRESHOLD = 50  # Threshold for determining if a scene is static

# Function to download videos using yt-dlp for multiple search terms
def download_videos(search_terms, num_results_per_term):
    # Define yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',  # Ensure videos are downloaded in MP4 format
        'noplaylist': True,  # Download only individual videos, not playlists
        'outtmpl': './videos/%(title)s.%(ext)s',  # Output template for downloaded videos
        'quiet': False,  # Display download progress
        'postprocessors': [],  # No post-processing
    }

    # Use yt-dlp to download videos based on search terms
    with ydl.YoutubeDL(ydl_opts) as ydl_instance:
        for term in search_terms:
            ydl_instance.download([f"ytsearch{num_results_per_term}:{term}"])

# Function to filter videos based on frame rate
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

# Function to extract metadata from a video and its scene
def extract_metadata(video_path, scene_path, x, y):
    # Extract video metadata
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    # Extract scene metadata
    scene_cap = cv2.VideoCapture(scene_path)
    scene_duration = int(scene_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    scene_start_frame = scene_cap.get(cv2.CAP_PROP_POS_FRAMES)
    scene_end_frame = scene_start_frame + scene_duration * fps
    scene_file_size = os.path.getsize(scene_path)
    scene_cap.release()

    # Return combined metadata
    return {
        'source_url': video_path,
        'fps': fps,
        'resolution': f"{width}x{height}",
        'duration': duration,
        'codec': codec,
        'x_value': x,
        'y_value': y,
        'scene_duration': scene_duration,
        'scene_start_frame': scene_start_frame,
        'scene_end_frame': scene_end_frame,
        'scene_file_size': scene_file_size
    }

# Function to extract and split scenes from a video using pyscenedetect
def extract_and_split_scenes(video_path, output_dir):
    # Initialize video and scene managers
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))  # Adjust the threshold as needed
    video_manager.set_downscale_factor()  # Downscale video for faster processing
    video_manager.start()  # Start video manager
    scene_manager.detect_scenes(frame_source=video_manager)  # Detect scenes
    
    # Split video into individual scenes and save them
    scene_list = scene_manager.get_scene_list(video_path)
    split_video_ffmpeg(video_manager, scene_list, output_dir, video_name="scene_$SCENE_NUMBER.mp4")
    
    # Return paths of the split scenes
    return [os.path.join(output_dir, f"scene_{i}.mp4") for i in range(len(scene_list))]

# Function to check if a scene is static based on average frame difference
def is_static_scene(scene_path):
    cap = cv2.VideoCapture(scene_path)
    total_diff = 0
    prev_frame = None
    frame_count = 0

    # Calculate the difference between consecutive frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            total_diff += np.mean(diff)
            frame_count += 1

        prev_frame = frame

    avg_diff = total_diff / frame_count
    cap.release()
    # Return True if the average difference is below the threshold, indicating a static scene
    return avg_diff < THRESHOLD

# Function to process each scene
def process_scene(scene_path, x, y):
    # Check if the scene is static
    if is_static_scene(scene_path):
        logging.info(f"Discarding static scene: {scene_path}")
        os.remove(scene_path)  # Remove the static scene
        return

    # Extract frames based on x and y values
    frames = extract_frames(scene_path, x, y)
    
    # Combine extracted frames into a single image
    combined_image = combine_frames(frames)
    
    # Organize the combined image into dataset folders
    organize_images(combined_image, x, y)

# Function to extract frames from a scene based on x and y values
def extract_frames(scene_path, x, y):
    cap = cv2.VideoCapture(scene_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frames = []

    # Extract every x frames
    for i in range(0, total_frames, x):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)
    
    # Extract y frames around each x frame
    y_frames = []
    for frame_index in range(len(extracted_frames)):
        start = max(frame_index - y // 2, 0)
        end = min(frame_index + y // 2, len(extracted_frames) - 1)
        for j in range(start, end + 1):
            y_frames.append(extracted_frames[j])

    cap.release()
    # Return the combined list of x and y frames
    return extracted_frames + y_frames

# Function to combine frames into a single image
def combine_frames(frames):
    # Check if there's an odd number of frames and handle accordingly
    if len(frames) % 2 != 0:
        frames.append(np.zeros_like(frames[-1]))  # Add a black frame to make it even
    return cv2.hconcat(frames)

# Function to create metadata for each dataset
def create_metadata(video_path, scene_path, x, y):
    metadata = extract_metadata(video_path, scene_path, x, y)
    return metadata  # Return the metadata

# Function to organize combined images into dataset folders
def organize_images(image, x, y):
    folder_name = f"x_{x}_y_{y}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    cv2.imwrite(os.path.join(folder_name, f"combined_{x}_{y}.jpg"), image)

# Main function to execute the entire process
def main():
    try:
        # Get user input for search terms and number of results
        search_terms = input("Enter the search terms separated by commas: ").split(',')
        num_results = int(input("Enter the number of results to fetch per term: "))

        # Download videos based on search terms
        download_videos(search_terms, num_results)

        # Filter downloaded videos based on FPS
        valid_videos = filter_videos_by_fps('./videos')

        metadata_list = []

        # Process each valid video with progress monitoring
        for video in tqdm(valid_videos, desc="Processing videos"):
            video_path = os.path.join('./videos', video)
            
            # Extract metadata for the video
            video_metadata = extract_metadata(video_path, None, None, None)
            metadata_list.append(video_metadata)

            # Split video into scenes
            scenes = extract_and_split_scenes(video_path, './scenes')
            for scene in scenes:
                # Check if the scene is static
                if is_static_scene(scene):
                    logging.info(f"Discarding static scene: {scene}")
                    os.remove(scene)  # Remove the static scene
                    continue

                # Process each non-static scene
                for x in [2, 4, 8, 16]:  # Example 'x' values
                    for y in [2, 4, 8, 16]:  # Example 'y' values
                        process_scene(scene, x, y)
                        scene_metadata = create_metadata(video_path, scene, x, y)
                        metadata_list.append(scene_metadata)

        # Save metadata to a JSON file
        with open('metadata.json', 'w') as f:
            json.dump(metadata_list, f)
        
        logging.info(f"Successfully processed {len(valid_videos)} videos.")
    except Exception as e:
        logging.error(f"Error encountered: {str(e)}")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()