import cv2
import pandas as pd
from face_recognition.face_recognition.face_landmarks import get_coords
from pathlib import Path
from typing import Tuple

PATH_TO_DATA = Path('../data/external/faces_actors_and_actresses.csv')
PATH_TO_IMAGE_FACES_ACTORS_AND_ACTRESS = Path('../data/external/images_actors_and_actresses_full')


def list_image_bbox() -> [Tuple]:
    """
    Calculates the coordinates of faces for each picture and writes them to a sheet
    Returns:
        List with names of image and coordinates faces of them
    """
    coordinates_faces_on_image = []
    data = pd.read_csv(str(PATH_TO_DATA))
    for ind, row in data.iterrows():
        name_image = row['name_image']
        image = cv2.imread(str(PATH_TO_IMAGE_FACES_ACTORS_AND_ACTRESS / f'{name_image}.jpg'))
        mas = get_coords(image)
        tpl = (name_image, mas)
        coordinates_faces_on_image.append(tpl)
    return coordinates_faces_on_image




