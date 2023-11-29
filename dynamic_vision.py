from typing import List, Tuple
import cv2
import numpy as np
import os
from advertise import ImageGenerator
from analytics import AnalyzeDynamicVision
from copy import deepcopy
from PIL import Image
from darknet import DarkNet
from resnet import Resnet50
import yaml


def decide_strategy():
    with open("config.yaml") as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
        if result["model"] == "darknet":
            return DarkNet()
        elif result["model"] == "resnet50":
            return Resnet50()
        else:
            raise NotImplementedError()


class DynamicVision:
    def __init__(self, enable_ads=True, enable_multi_object_detection=True) -> None:
        self.input_folder_path = "input_videos"
        self.output_folder_path = "input_videos"
        self.image_generator = ImageGenerator()
        self.enable_ads = enable_ads
        self.enable_multi_object_detection = enable_multi_object_detection
        self.original_video = None
        self.model_strategy = decide_strategy()
        self.pred_dict = {}
        self.analyzer = AnalyzeDynamicVision()

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

    def generate_ads(self, video_results: List[Tuple]):
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

    def __write_prediction_to_image(self, input_image, prediction):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (0, 0, 255)
        thickness = 2
        text_size = cv2.getTextSize(prediction, font, scale, thickness)[0]
        text_width, text_height = text_size
        _, img_width = input_image.shape[:2]
        text_x = img_width - text_width - 20
        text_y = text_height + 20
        return cv2.putText(
            input_image, prediction, (text_x, text_y), font, scale, color, thickness
        )

    def write_ads_to_video(self, ad_result):
        ad_generated_video = deepcopy(self.original_video)
        new_video = []
        for frame_no, frame_result in enumerate(ad_result):
            _, prediction, _, ad_file_path = frame_result

            if ad_file_path:
                advertisement = self.__read_image(ad_file_path)
                new_frame = self.__overwrite_image(
                    ad_generated_video[frame_no], advertisement
                )
            else:
                new_frame = ad_generated_video[frame_no]
            if prediction:
                new_frame_with_text = self.__write_prediction_to_image(
                    new_frame, prediction
                )
            else:
                new_frame_with_text = new_frame
            new_video.append(new_frame_with_text)
        return new_video

    def __save_ad_video(self, output_video, input_file_path):
        height, width = output_video[0].shape[:2]
        file_name_without_extension = input_file_path[:-4]
        video = cv2.VideoWriter(
            f"{file_name_without_extension}_advertisement.avi",
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
        self.original_video = input_video
        classification_result, _ = self.model_strategy.classify_objects(input_video)

        if self.enable_ads:
            ad_result = self.generate_ads(classification_result)
            new_video = self.write_ads_to_video(ad_result)
            self.__save_ad_video(new_video, input_file_path)
        if self.enable_multi_object_detection:
            Resnet50().classify_objects_in_video(input_file_path)

    def analyze(self, input_file_path):
        # Capture the input video
        input_video = self.capture_video(input_file_path)
        self.original_video = input_video
        _, all_classification = Resnet50().classify_objects(input_video)
        self.analyzer.analyze(all_classification)
