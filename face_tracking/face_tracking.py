from __future__ import division
import os
import cv2
from PIL import Image
import face_recognition
from .eye import Eye
from .calibration import Calibration
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from config import Config
from .hopenet import Hopenet
from .utils import *


class FaceTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self, config=Config):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.face_bounds = None
        self.config = config
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = face_recognition.face_locations

        # _predictor is used to get facial landmarks of a given face
        self._predictor = face_recognition.face_landmarks

        # Load model for head pose estimation
        self.head_pose = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/hopenet_robust_alpha1.pkl"))
        saved_state_dict = torch.load(model_path, map_location="cpu")
        self.head_pose.load_state_dict(saved_state_dict, strict=False)
        self.head_pose.eval()

        # Preprocessing required for head pose estimation
        self.transformations = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    @property
    def face_located(self):
        """Check that the face has been located"""
        return self.face_bounds is not None

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        
        # Resize image while maintaining aspect ratio (to increase face detection speed)
        original_size = self.frame.shape[:2]
        frame_face_detect = resize_image(self.frame, 256)
        size_factor = min(original_size) / min(frame_face_detect.shape[:2])

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        gray_frame = cv2.cvtColor(self.frame.copy(), cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_face_detect = cv2.cvtColor(frame_face_detect, cv2.COLOR_BGR2RGB)

        face_locs = self._face_detector(frame_face_detect, model=self.config.face_detection_model)
        self.num_faces = len(face_locs)

        self.face_bounds = None if len(face_locs) == 0 else face_locs[0]
        # Use size_factor to extrapolate detected face locations to the original sized frame
        if self.face_bounds is not None:
            self.face_bounds = tuple(int(i * size_factor) for i in self.face_bounds)

        if self.face_located:
            
            ### Gaze tracking
            
            try:
                landmarks = self._predictor(frame, face_locations=[self.face_bounds])[0]
                self.eye_left = Eye(gray_frame, landmarks, 0, self.calibration)
                self.eye_right = Eye(gray_frame, landmarks, 1, self.calibration)

            except IndexError:
                self.eye_left = None
                self.eye_right = None

            ### Head pose estimation

            eval_face = frame[max(0, self.face_bounds[0] - 100):min(frame.shape[0], self.face_bounds[2] + 100),
                                max(0, self.face_bounds[3] - 100):min(frame.shape[1], self.face_bounds[1] + 100)]
            # im = Image.fromarray(eval_face)
            # im.save("your_file.jpeg")
            eval_face = self.transformations(Image.fromarray(eval_face))

            eval_face_shape = eval_face.size()
            eval_face = eval_face.view(1, eval_face_shape[0], eval_face_shape[1], eval_face_shape[2])

            yaw, pitch, roll = self.head_pose(eval_face)
            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)
            # Get continuous predictions in degrees.
            self.yaw_predicted = (torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99).data.numpy().item()
            self.pitch_predicted = (torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99).data.numpy().item()
            self.roll_predicted = (torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99).data.numpy().item()

            # Reverse sign of roll to ensure it is +ve when head is towards right (as yaw is also +ve for right)
            self.roll_predicted = -self.roll_predicted

        else:
            self.yaw_predicted, self.pitch_predicted, self.roll_predicted = None, None, None
            self.eye_left, self.eye_right = None, None


    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / self.eye_left.width
            pupil_right = self.eye_right.pupil.x / self.eye_right.width
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / self.eye_left.height
            pupil_right = self.eye_right.pupil.y / self.eye_right.height
            return (pupil_left + pupil_right) / 2

    def eyes_are_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.4

    def eyes_are_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.6

    def eyes_are_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.eyes_are_right() is not True and self.eyes_are_left() is not True

    def eyes_are_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def face_check(self):
        """Check for multiple faces / face not visible conditions"""
        if self.num_faces == 0:
            return "Face not visible"
        elif self.num_faces > 1:
            return "Multiple people detected"

    def face_is_looking_away(self):
        """Check if the person is looking away acc to defined thresholds"""
        if self.face_located:
            return (abs(self.yaw_predicted) > 25) | (abs(self.pitch_predicted) > 20)

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        if self.face_located:
            x_min, x_max = self.face_bounds[3], self.face_bounds[1]
            y_min, y_max = self.face_bounds[0], self.face_bounds[2]

            cv2.rectangle(frame, 
                (x_min - 10, y_min - 30), 
                (x_max + 10, y_max + 20), 
                (0,255,0), 1)
            draw_axis(frame, self.yaw_predicted, self.pitch_predicted, self.roll_predicted,
             tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = abs(y_max - y_min)/2)

        return frame
