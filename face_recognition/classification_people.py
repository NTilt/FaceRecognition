import pickle
import argparse
import cv2
from pathlib import Path
from main import get_face_coordinates, get_features_from_image_face


def classification_people_on_image(path_to_image: Path, path_to_kdtree: Path):
    """
    Print the index of the photo with the image of a person in the input photo
    Args:
        path_to_image: Path to image
        path_to_kdtree: Path to encoding kdtree

    """
    image = cv2.imread(str(path_to_image))
    faces = []
    with open(str(path_to_kdtree), "rb") as file:
        kdtree = pickle.load(file)
    rects = get_face_coordinates(image)
    for face_coordinate in rects:
        feature = get_features_from_image_face(image, face_coordinate).reshape(1, -1)
        dist, ind = kdtree.query(feature)
        faces.append({'index': ind[0][0], 'probability': 100 - dist[0][0] * 100})
    for item in faces:
        print(item)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to image for classifications people",
                    type=Path)
parser.add_argument("-db", "--database", help="Path to kdtree",
                    default=Path("face_recognition/data/processed/features_index.pkl"))
args = parser.parse_args()

classification_people_on_image(path_to_image=args.image,
                               path_to_kdtree=args.database)