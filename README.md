# dynamic-vision

Steps: 
1. Install requirements
    - `pip install -r requirements.txt`
2. Download models. 
    - Required models: 
        - [Resnet50](https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/resnet50-19c8e357.pth/)
        - [YoloV3](https://pjreddie.com/media/files/yolov3.weights) weights for training `Darknet`
        - [yolov3.pt](https://drive.google.com/file/d/11PcWtxLRIaofSwIIi1ZxYh9RgpIviB2o/view?usp=share_link)
3. Place input videos in `input_videos` folder
4. Place downloaded models in `models` folder
5. Add OpenAI API key to `.env` file as per the `.env.example` file provided
5. Run `python darknet_model.py`. Once the script has run successfully, you will see `darknet.h5` in `models` folder.
6. Activate your `Dynamic Vision` using `python main.py`


Output consists of 3 parts:
1. `Multiobject Detection` : Stored in `output_videos` folder
2. `Advertisement Generation` : Stored in root folder
3. `Analytics` : Frame by Frame analytics stored 


`config.yaml` has data that helps to change the configurations of the program.

| Configuration |         |          |   |   |
|---------------|---------|----------|---|---|
| Model         | darknet | resnet50 |   |   |
| Chart type    | bar     | pie      |   |   |
|               |         |          |   |   |