from imageai.Detection import VideoObjectDetection
import os
from imageai.Classification import ImageClassification
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
        vid_obj_detect.detectObjectsFromVideo(
            input_file_path=os.path.join(self.input_folder_path, input_video_path),
            output_file_path=os.path.join(
                self.output_folder_path, f"{input_video_path[:-4]}_output"
            ),
            frames_per_second=40,
            log_progress=True,
            return_detected_frame=True,
        )

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
        video_results_best_prediction = []
        video_results_top_n_predictions = []

        for frame_no, frame in enumerate(input_frames):
            image_result = self.classify_object_in_image(frame, threshold=0)
            best_prediction = self.__get_highest_probability_prediction(image_result)
            all_predictions = self.__get_top_n_predictions(image_result)
            if best_prediction:
                current_prediction, current_probability = best_prediction[0]
            else:
                current_prediction, current_probability = None, None
            video_results_best_prediction.append(
                (frame_no, current_prediction, current_probability)
            )
            video_results_top_n_predictions.append((frame_no, all_predictions))

        return video_results_best_prediction, video_results_top_n_predictions
