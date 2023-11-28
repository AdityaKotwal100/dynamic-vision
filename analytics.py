import matplotlib.pyplot as plt
import yaml
import os


def get_chart_type():
    with open("config.yaml") as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
        return result["chart_type"]


class AnalyzeDynamicVision:
    def __init__(self) -> None:
        self.chart_type = get_chart_type()
        self.analytics_path = "analytics"

    def analyze(self, input_video_data):
        if not os.path.exists(self.analytics_path):
            os.mkdir(self.analytics_path)
        for data in input_video_data:
            if self.chart_type == "bar":
                frame_no, model_inference = data
                labels = [item[0] for item in model_inference]
                values = [item[1] for item in model_inference]
                plt.barh(labels, values, color="skyblue")
            elif self.chart_type == "pie":
                frame_no, model_inference = data
                labels = [item[0] for item in model_inference]
                values = [item[1] for item in model_inference]
                max_index = values.index(max(values))

                zoom_size = [0] * len(values)
                zoom_size[max_index] = 0.2
                plt.pie(values, labels=labels, explode=zoom_size)
            # Save the chart image
            chart_filename = f"frame_{frame_no + 1}_{self.chart_type}_chart.png"
            plt.savefig(os.path.join(self.analytics_path, chart_filename))
            plt.close()  # Close the plot to avoid displaying it
