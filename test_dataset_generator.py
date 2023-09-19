import unittest
import os
import cv2
import shutil
from dataset_generator import download_videos, filter_videos_by_fps, extract_metadata

class TestDatasetGenerator(unittest.TestCase):

    def setUp(self):
        # This method will run before each test
        self.search_terms = ["short sample video"]
        self.num_results = 1
        self.test_dir = "./test_videos"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def test_download_videos(self):
        download_videos(self.search_terms, self.num_results)
        self.assertTrue(os.listdir('./videos'))  # Check if videos directory has content

    def test_filter_videos_by_fps(self):
        valid_videos = filter_videos_by_fps('./videos')
        self.assertTrue(valid_videos)  # Check if there are valid videos
        for video in valid_videos:
            cap = cv2.VideoCapture(video)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.assertEqual(fps, 30)  # Check if video is 30fps
            cap.release()

    def test_extract_metadata(self):
        video_path = os.path.join(self.test_dir, os.listdir(self.test_dir)[0])  # Get the first video
        metadata = extract_metadata(video_path, None, None, None)
        self.assertIn('fps', metadata)
        self.assertIn('resolution', metadata)
        self.assertIn('duration', metadata)
        self.assertEqual(metadata['fps'], 30)

    # ... Additional refined tests for other functions ...

    def tearDown(self):
        # This method will run after each test to clean up
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
