# create.py

import json
import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import json
from base64 import b64decode
from pathlib import Path
import time

load_dotenv()


class ImageGenerator:
    def __init__(self):
        # self.PROMPT = 'A banner with text in English providing advertisements offering promotions for a "{}"'
        self.PROMPT = r"An advertisement for a {}"
        self.counter = 0

    def check_counter(self):
        self.counter += 1
        return self.counter >= 5

    def generate_image(self, object):
        if self.check_counter():
            print("Limit reached!")
            return ""

        current_prompt = self.PROMPT.format(object)
        DATA_DIR = Path.cwd() / "responses"

        DATA_DIR.mkdir(exist_ok=True)

        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = None

        response = openai.Image.create(
            prompt=current_prompt,
            n=1,
            size="256x256",
            response_format="b64_json",
        )
        while not response:
            print("Sleeping..")
            time.sleep(1)

        json_file_name = f"{current_prompt[:5]}-{response['created']}.json"
        file_name = DATA_DIR / json_file_name

        with open(file_name, mode="w", encoding="utf-8") as file:
            json.dump(response, file)

        return self.__convert_to_image(json_file_name)

    def __convert_to_image(self, json_file_name):
        DATA_DIR = Path.cwd() / "responses"
        JSON_FILE = DATA_DIR / json_file_name
        IMAGE_DIR = Path.cwd() / "images" / JSON_FILE.stem

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        with open(JSON_FILE, mode="r", encoding="utf-8") as file:
            response = json.load(file)

        for index, image_dict in enumerate(response["data"]):
            image_data = b64decode(image_dict["b64_json"])
            image_file = IMAGE_DIR / f"{JSON_FILE.stem}-{index}.png"
            with open(image_file, mode="wb") as png:
                png.write(image_data)

        return image_file


if __name__ == "__main__":
    d = ImageGenerator()
    d.generate_image("Car")
