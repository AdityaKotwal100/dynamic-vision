from dynamic_vision import DynamicVision
import os


def check_prerequisites():
    required_folders = ["input_videos", "output_videos", "models"]
    for folder in required_folders:
        if not os.path.exists(folder):
            raise Exception(f"{folder} folder missing")


check_prerequisites()
d = DynamicVision(enable_ads=False)
# d.activate(input_file_path="home.mp4")
d.analyze(input_file_path="home.mp4")
