# Frame-Combo-Gen

Frame-Combo-Gen is a tool designed to generate datasets from YouTube videos based on specified 'x' and 'y' values. It extracts frames from videos, combines them into single images, and organizes them into structured datasets.

## Features:

- **Video Searching and Downloading**: Uses `yt-dlp` to search for and download videos from YouTube based on user-specified search terms.
- **Metadata Extraction**: Extracts metadata from each downloaded video, including frame rate, resolution, codec, and duration.
- **Frame Rate Filtering**: Processes only videos with a frame rate of 30fps.
- **Scene Detection and Splitting**: Uses `PySceneDetect` to split videos into individual scenes.
- **Frame Extraction**: Extracts frames from each scene based on 'x' and 'y' values, both of which are powers of 2.
- **Static Scene Detection and Removal**: Removes scenes that are relatively static or have minimal changes.
- **Frame Combination**: Combines extracted frames into single images.
- **Dataset Organization**: Organizes combined images into folders based on their 'x' and 'y' values.
- **Dataset Metadata Creation**: Collects and structures metadata for each dataset.
- **Progress Monitoring**: Provides real-time feedback on the dataset generation process with a progress bar.
- **Logging**: Logs the process and any issues for troubleshooting and reference.

## Usage:

1. Ensure you have all the required libraries installed:
   ```bash
   pip install yt-dlp pyscenedetect cv2 tqdm
   ```

2. Run the script:
   ```bash
   python frame_combo_gen.py
   ```

3. Follow the on-screen prompts to specify search terms and the number of results to fetch.

4. The script will process the videos and generate datasets based on the specified 'x' and 'y' values.

## Contributing:

Contributions to Frame-Combo-Gen are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.

## License:

GPL V3