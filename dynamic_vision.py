from typing import List
import cv2
import numpy as np
import os
from advertise import ImageGenerator
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
from darknet import DarkNet
from resnet import Resnet50
import yaml


def decide_strategy():
    with open("model.yaml") as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
        if result["model"] == "darknet":
            return DarkNet()
        elif result["model"] == "resnet50":
            return Resnet50()


class DynamicVision:
    def __init__(self, write_video=True, enable_ads=True) -> None:
        self.input_folder_path = "input_videos"
        self.output_folder_path = "output_videos"
        self.model_folder_path = "models"
        self.model = None
        self.image_generator = ImageGenerator()
        self.enable_ads = enable_ads
        self.original_video = None
        self.model_strategy = decide_strategy()
        self.pred_dict = {}

    def capture_video(self, file_name: str) -> List[np.ndarray]:
        video_frames = []
        cap = cv2.VideoCapture(os.path.join(self.input_folder_path, file_name))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(frame)

        cap.release()
        cv2.destroyAllWindows()

        return video_frames

    def generate_ads(self, video_results):
        ## Prediction could be none
        _, previous_prediction, _ = video_results[0]
        current_result = video_results[0]
        previous_ad_path = ""
        if not previous_prediction:
            ad_file_path = ""
        else:
            ad_file_path = self.image_generator.generate_image(previous_prediction)
            previous_ad_path = ad_file_path
            self.pred_dict[previous_prediction] = ad_file_path
        new_result = current_result + (ad_file_path,)
        video_results[0] = new_result
        ad_file_path = ""
        for result_idx in range(1, len(video_results)):
            _, current_prediction, _ = video_results[result_idx]
            if previous_prediction != current_prediction:
                current_result = video_results[result_idx]
                if not current_prediction:
                    # ad_file_path = ""
                    ad_file_path = self.pred_dict[previous_prediction]
                elif current_prediction in self.pred_dict:
                    ad_file_path = self.pred_dict[current_prediction]
                else:
                    ad_file_path = self.image_generator.generate_image(
                        current_prediction
                    )
                    self.pred_dict[current_prediction] = ad_file_path
                previous_ad_path = ad_file_path
            else:
                ad_file_path = previous_ad_path

            new_result = current_result + (ad_file_path,)
            video_results[result_idx] = new_result
            previous_prediction = current_prediction
        return video_results

    def __read_image(self, file_path):
        return np.array(Image.open(str(file_path)))

    def __overwrite_image(self, original_frame, advertisement):
        height1, width1, _ = original_frame.shape
        height2, width2, _ = advertisement.shape
        start_height = height1 - height2
        start_width = width1 - width2
        start_height = max(0, start_height)
        start_width = max(0, start_width)
        original_frame[start_height:height1, start_width:width1] = advertisement
        return original_frame

    def write_ads_to_video(self, ad_result):
        ad_generated_video = deepcopy(self.original_video)
        new_video = []
        for frame_no, frame_result in enumerate(ad_result):
            _, _, _, ad_file_path = frame_result

            if ad_file_path:
                advertisement = self.__read_image(ad_file_path)
                new_frame = self.__overwrite_image(
                    ad_generated_video[frame_no], advertisement
                )
            else:
                new_frame = ad_generated_video[frame_no]
            new_video.append(new_frame)
        return new_video

    def __save_ad_video(self, output_video):
        height, width = output_video[0].shape[:2]
        video = cv2.VideoWriter(
            "final.avi",
            cv2.VideoWriter_fourcc(*"DIVX"),
            10.0,
            (width, height),
            isColor=True,
        )
        for image in output_video:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

    def activate(self, input_file_path):
        # Capture the input video
        input_video = self.capture_video(input_file_path)
        input_video = input_video[:15]
        self.original_video = input_video
        classification_result = self.model_strategy.classify_objects(input_video)

        if self.enable_ads:
            ad_result = self.generate_ads(classification_result)
            new_video = self.write_ads_to_video(ad_result)
            self.__save_ad_video(new_video)
        Resnet50().classify_objects_in_video(input_file_path)
