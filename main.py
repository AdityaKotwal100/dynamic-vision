from dynamic_vision import DynamicVision
import os


def check_prerequisites():
    required_folders = ["input_videos", "output_videos", "models"]
    for folder in required_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


check_prerequisites()
input_file_name = "home.mp4"
dynamic_vision = DynamicVision()
dynamic_vision.activate(input_file_path=input_file_name)
dynamic_vision.analyze(input_file_path=input_file_name)
