import subprocess
from typing import List
subprocess.call(['/model/venv/bin/python3', '-m', 'pip', 'install', 'opencv-python-headless'])

import requests
from io import BytesIO
import itertools
import numpy as np
import mediapipe as mp

import base64
from PIL import Image

from nbox import operator

@operator()
class MediaPipeModel():
  def __init__(self):
    mp_pose = mp.solutions.pose
    self.pose = mp_pose.Pose(
      static_image_mode=True, model_complexity=2, min_detection_confidence=0.8
    )

  def predict(self, image_array):
    mp_pose = mp.solutions.pose
    landmarks = [
      "LEFT_ANKLE","LEFT_EAR","LEFT_ELBOW","LEFT_EYE","LEFT_EYE_INNER","LEFT_EYE_OUTER","LEFT_FOOT_INDEX",
      "LEFT_HEEL","LEFT_HIP","LEFT_INDEX","LEFT_KNEE","LEFT_PINKY","LEFT_SHOULDER","LEFT_THUMB","LEFT_WRIST",
      "MOUTH_LEFT","MOUTH_RIGHT","NOSE","RIGHT_ANKLE","RIGHT_EAR","RIGHT_ELBOW","RIGHT_EYE","RIGHT_EYE_INNER",
      "RIGHT_EYE_OUTER","RIGHT_FOOT_INDEX","RIGHT_HEEL","RIGHT_HIP","RIGHT_INDEX","RIGHT_KNEE","RIGHT_PINKY",
      "RIGHT_SHOULDER","RIGHT_THUMB","RIGHT_WRIST",
    ]
    coordinates = ["x", "y", "z", "visibility"]

    data = {}
    image = np.array(image_array).astype(np.uint8)
    image_height, image_width, _ = image.shape
    data["image_width"] = image_width
    data["image_height"] = image_height
    results = self.pose.process(image)
    if results.pose_landmarks:
      for l, c in itertools.product(landmarks, coordinates):
        data[f"{l}_{c}"] = results.pose_landmarks.landmark[mp_pose.PoseLandmark[l]].__getattribute__(c)
    return {"pred": data}

  def predict_b64(self, image_b64: str, shape: List[int]):
    img = Image.frombytes("RGB", shape, base64.b64decode(image_b64))
    image_array = np.array(img, dtype = np.uint8).reshape(*shape, 3)
    return self.predict(image_array)

  def predict_url(self, url: str):
    r = requests.get(url)
    img = Image.open(BytesIO(r.content))
    return self.predict(np.array(img))
