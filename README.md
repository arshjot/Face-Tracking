# Face Tracking

A face tracking system which includes eye tracking as well as head pose estimation. It has been developed by combining
and modifying these two libraries - https://github.com/antoinelame/GazeTracking 
and https://github.com/natanielruiz/deep-head-pose

## Steps to run

1. Run the below command to install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Download the head pose estimation model weights from 
[here](https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR), extract, and place it under the 
`trained_models` directory. Ensure that the below directory structure is followed:
    ```bash
    ├── face_tracking
    │   ├── trained_models
    │   │   ├── hopenet_robust_alpha1.pkl
    │   │   └── .keep
    ```
3. If required, modify the `config.py` file to change input stream source and face detection model.
4. Run the face tracking system:
    ```
    python test.py
    ```

Sample output - https://drive.google.com/file/d/13XoT8W9SNqG5t6ZfJXskclFRZyDi5jF5/view?usp=sharing

## Citations

1. [Antoine Lamé's](https://github.com/antoinelame) [Gaze Tracking library](https://github.com/antoinelame/GazeTracking)
2. [Nataniel Ruiz's](https://github.com/natanielruiz) [Deep Head Pose Esimation library](https://github.com/natanielruiz/deep-head-pose)
