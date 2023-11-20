from typing import List
import cv2
import numpy as np
from imageai.Detection import VideoObjectDetection
import os
from imageai.Classification import ImageClassification
from advertise import ImageGenerator
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt


class DynamicVision:
    def __init__(self, write_video=True, enable_ads=True) -> None:
        self.input_folder_path = "input_videos"
        self.output_folder_path = "output_videos"
        self.model_folder_path = "models"
        self.model = None
        self.predictor = ImageClassification()
        self.image_generator = ImageGenerator()
        self.write_video = write_video
        self.enable_ads = enable_ads
        self.original_video = None

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

    def classify_objects_in_video(self, input_video_path: str):
        if not self.model:
            raise Exception("Model not set. Set model using obj.set_model()")
        vid_obj_detect = VideoObjectDetection()
        vid_obj_detect.setModelTypeAsYOLOv3()
        vid_obj_detect.setModelPath(os.path.join(self.model_folder_path, self.model))
        vid_obj_detect.loadModel()
        detected_vid_obj = vid_obj_detect.detectObjectsFromVideo(
            input_file_path=os.path.join(self.input_folder_path, input_video_path),
            output_file_path=os.path.join(
                self.output_folder_path, f"{input_video_path[:-4]}_output"
            ),
            frames_per_second=40,
            log_progress=True,
            return_detected_frame=True,
        )

    def set_model(self, model: str):
        if not os.path.exists(os.path.join(self.model_folder_path, model)):
            raise Exception("Model not present in directory")

        self.model = model

    def __get_highest_probability_prediction(self, prediction_results):
        max_result = []
        max_prob_so_far = 0
        for result in prediction_results:
            prediction, probability = result
            if probability > max_prob_so_far:
                max_result = [(prediction, probability)]
                max_prob_so_far = probability

        return max_result

    def __get_top_n_predictions(self, prediction_results, n=10):
        return sorted(prediction_results, key=lambda x: x[0])[:n]

    def classify_object_in_image(self, image, threshold=80):
        predictions, probabilities = self.predictor.classifyImage(
            image, result_count=10
        )
        return [
            (eachPrediction, eachProbability)
            for eachPrediction, eachProbability in zip(predictions, probabilities)
            if eachProbability >= threshold
        ]

    def initialize_model(self):
        self.predictor.setModelTypeAsResNet50()
        self.predictor.setModelPath(
            os.path.join(self.model_folder_path, "resnet50-19c8e357.pth")
        )
        self.predictor.loadModel()

    def classify_objects(self, input_frames):
        video_results = []
        for frame_no, frame in enumerate(input_frames):
            image_result = self.classify_object_in_image(frame)
            if best_prediction := self.__get_highest_probability_prediction(
                image_result
            ):
                current_prediction, current_probability = best_prediction[0]
            else:
                current_prediction, current_probability = None, None
            video_results.append((frame_no, current_prediction, current_probability))

        return video_results

    def generate_ads(self, video_results):
        ## Prediction could be none
        _, previous_prediction, _ = video_results[0]
        current_result = video_results[0]
        previous_ad_path = ""
        # TODO: Check the Condition RHS
        if not previous_prediction:
            ad_file_path = ""
        else:
            ad_file_path = self.image_generator.generate_image(previous_prediction)
            previous_ad_path = ad_file_path
        new_result = current_result + (ad_file_path,)
        video_results[0] = new_result
        ad_file_path = ""
        for result_idx in range(1, len(video_results)):
            _, current_prediction, _ = video_results[result_idx]
            if previous_prediction != current_prediction:
                current_result = video_results[result_idx]
                # TODO: Check the Condition RHS
                if not current_prediction:
                    ad_file_path = ""
                else:
                    ad_file_path = self.image_generator.generate_image(
                        current_prediction
                    )
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
        add_generated_video = deepcopy(self.original_video)
        new_video = []
        for frame_no, frame_result in enumerate(ad_result):
            _, _, _, ad_file_path = frame_result

            if ad_file_path:
                advertisement = self.__read_image(ad_file_path)
                new_frame = self.__overwrite_image(
                    add_generated_video[frame_no], advertisement
                )
            else:
                new_frame = add_generated_video[frame_no]
            new_video.append(new_frame)
        return new_video

    def __save_ad_video(self, output_video):
        height, width = output_video[0].shape[:2]
        video = cv2.VideoWriter(
            "final.avi",
            cv2.VideoWriter_fourcc(*"DIVX"),
            20.0,
            (width, height),
            isColor=True,
        )
        for image in output_video:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

    def activate(self, input_file_path):
        input_video = self.capture_video(input_file_path)
        self.original_video = input_video
        self.initialize_model()
        classification_result = self.classify_objects(input_video)
        if self.enable_ads:
            ad_result = self.generate_ads(classification_result)
            new_video = self.write_ads_to_video(ad_result)
            self.__save_ad_video(new_video)
