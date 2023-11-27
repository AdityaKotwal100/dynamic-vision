from imageai.Detection import VideoObjectDetection
import os
from imageai.Classification import ImageClassification
from advertise import ImageGenerator
import matplotlib.pyplot as plt
from strategy import ModelStrategy


class Resnet50(ModelStrategy):
    def __init__(self) -> None:
        self.input_folder_path = "input_videos"
        self.output_folder_path = "output_videos"
        self.model_folder_path = "models"
        self.model = "yolov3.pt"
        self.predictor = ImageClassification()

    def classify_objects_in_video(self, input_video_path: str):
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
                print(prediction)
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
        self.initialize_model()
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
