import cv2 as cv
import numpy as np
import mediapipe as mp
import os

OPEN_PATH = "D:/coding/ImageHandAnalysis/images"
SAVE_PATH = "D:/coding/ImageHandAnalysis/cropped"
MODELS_PATH = "D:/coding/ImageHandAnalysis/models"
OFFSET = 10

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=f'{MODELS_PATH}/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1,
    min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

for file in os.scandir(OPEN_PATH):
    if not file.is_file():
        continue

    image = mp.Image.create_from_file(file.path)
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    image_height, image_width, _ = image_copy.shape

    x_min = 0
    x_max = 224
    y_min = 0
    y_max = 224
    
    SAVE_FLAG = False
    
    for hand_landmarks in detection_result.hand_landmarks:
        SAVE_FLAG = True
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]

        x_min = max(int(min(xs) * image_width) - OFFSET, 0)
        y_min = max(int(min(ys) * image_height) - OFFSET, 0)
        x_max = min(int(max(xs) * image_width) + OFFSET, image_width - 1)
        y_max = min(int(max(ys) * image_height) + OFFSET, image_height - 1)

    if(SAVE_FLAG == False):
        continue

    width = x_max-x_min
    height = y_max-y_min
    dim = max(width, height)
    x_max = x_min+dim
    y_max = y_min+dim
    
    print(f'x: {x_min}, {x_max}; y: {y_min}, {y_max}')
    
    cropped_img = image_copy[y_min:y_max, x_min:x_max]
    resized_img = cv.resize(cropped_img, (224, 224), interpolation=cv.INTER_CUBIC)
    final_img = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
    
    filename = file.name.split('.')
    cv.imwrite(f'{SAVE_PATH}/{filename[0]}_cropped.{filename[1]}', final_img)
    
    # cv.imshow("img", final_img) 
    # cv.waitKey()
    # cv.destroyAllWindows()     
