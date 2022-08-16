from pathlib import Path
import dlib
import numpy as np
import cv2
import pandas as pd
import pickle
from hashlib import md5
from sklearn.neighbors import KDTree

PATH_TO_SMALL_IMAGE = Path('face_recognition/data/processed/small_images_with_face')
PATH_TO_BASE = Path('face_recognition/data/processed/base_faces.csv')
PATH_TO_PICKLE_KDTREE_OBJECT = Path('face_recognition/data/processed/features_index.pkl')
PATH_TO_MODEL = Path('face_recognition/models/dlib_face_recognition_resnet_model_v1.dat')
PATH_TO_PREDICTOR = Path('face_recognition/models/shape_predictor_68_face_landmarks.dat')
PATH_TO_FULL_DATA_FACES = Path('face_recognition/data/external/images_actors_and_actresses_full')
PATH_TO_IMAGE = Path('face_recognition/data/external/faces_actors_and_actresses.csv')


def create_database(path_to_csv_file_with_image: Path,
                    path_to_full_data_faces=PATH_TO_FULL_DATA_FACES,
                    path_to_pickle_object=PATH_TO_PICKLE_KDTREE_OBJECT,
                    path_to_base=PATH_TO_BASE):
    """
    This function updates the people face database containing face_id, face_coordinates and face_features and
    save all information to csv file
    Args:
        path_to_csv_file_with_image: File path containing images of people and their bbox coordinates
        path_to_full_data_faces: File path to image
        path_to_pickle_object: File path to save pickle kdtree object
        path_to_base: File path to Full DataBase

    """
    x_features = []
    final_data = []
    data = pd.read_csv(str(path_to_csv_file_with_image))
    for ind, row in data.iterrows():
        name_image = row['name_image']
        path_to_image = (path_to_full_data_faces / f'{name_image}.jpg')
        image = cv2.imread(str(path_to_image))
        rects = get_face_coordinates(image)
        for face_coordinate in rects:
            x1 = face_coordinate.left()
            y1 = face_coordinate.top()
            x2 = face_coordinate.right()
            y2 = face_coordinate.bottom()
            cropped_image = get_cropped_image_face(image, face_coordinate)
            features = get_features_from_image_face(image, face_coordinate)
            face_id = get_face_id(cropped_image)
            final_data.append({'image_name': name_image, 'face_id': face_id, 'face_coord_x1': x1,
                               'face_coord_x2': x2, 'face_coord_y1': y1, 'face_coord_y2': y2})

            x_features.append(features)
        tree = KDTree(x_features)
        with open(str(path_to_pickle_object), 'wb') as file:
            pickle.dump(tree, file)
        df = pd.DataFrame(final_data)
        df.to_csv(str(path_to_base))


def get_face_id(image: np.ndarray) -> str:
    """
    Get unique id corresponding to each face
    Args:
        image: Image with faces of people

    Returns:
        face_id: Unique id corresponding to each face
    """
    face_id = md5(image).hexdigest()[0:4]
    return face_id


def get_cropped_image_face(image: np.ndarray, face_coordinates: dlib.rectangle) -> np.ndarray:
    """
    This function returns the cropped image of faces
    Args:
        image: 3d-matrix from original image
        face_coordinates: Coordinates of people's faces in the image

    Returns:
        3d-matrix from cropped image
    """
    shape_predictor = dlib.shape_predictor(str(PATH_TO_PREDICTOR))
    shapes_predictor = shape_predictor(image=image, box=face_coordinates)
    cropped_image = dlib.get_face_chip(image, shapes_predictor)
    return cropped_image


def get_features_from_image_face(img: np.ndarray, face_coordinates: dlib.rectangle,
                                 path_to_model=PATH_TO_MODEL,
                                 path_to_predictor=PATH_TO_PREDICTOR) -> np.ndarray:
    """
    Getting features from face.

    Args:
        img: Image from which you need to collect information
        face_coordinates: The coordinate of the person's face from the image
        path_to_model: Path to face recognitions model
        path_to_predictor: Path to predictor

    Returns:
        features: An array of values characterizing each person's face
    """
    face_recognition_model = dlib.face_recognition_model_v1(str(path_to_model))
    shape_predictor = dlib.shape_predictor(str(path_to_predictor))
    shapes_predictor = shape_predictor(image=img, box=face_coordinates)
    features = np.array(face_recognition_model.compute_face_descriptor(img, shapes_predictor))
    return features


def get_face_coordinates(image: np.array) -> dlib.rectangles:
    """
    This function returns all coordinates of the faces in this photo.
    Args:
        image: Image in the form of a 3D matrix

    Returns:
        rects: The dlib library object containing information about the coordinates of the faces in the photo
    """

    detector = dlib.get_frontal_face_detector()
    rects = detector(image)
    return rects


if __name__ == "__main__":
    create_database(PATH_TO_IMAGE)

