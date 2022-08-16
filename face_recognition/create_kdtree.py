import argparse
import pandas as pd
from pathlib import Path
import cv2
from main import get_features_from_image_face, get_cropped_image_face
from dlib import rectangle
from sklearn.neighbors import KDTree
import pickle


def create_kdtree(path_to_csv: Path, path_to_images_from_csv_files: Path) -> KDTree:
    """
    Args:
        path_to_csv: Csv-file or path to directory with csv-files.
        path_to_images_from_csv_files: Path to images from csv-files.

    Returns:
        KDtree: Built KDTree.
    """
    directory = True
    if path_to_csv.is_file():
        directory = False
    if directory:
        all_files = [i for i in path_to_csv.glob('*.csv')]
        final_data = pd.concat([pd.read_csv(f) for f in all_files])
    else:
        final_data = pd.read_csv(str(path_to_csv))
    features = []
    for ind, row in final_data.iterrows():
        name_image = row['image_name']
        path_to_image = (str(path_to_images_from_csv_files / f'{name_image}.jpg'))
        image = cv2.imread(str(path_to_image))
        x1 = row['face_coord_x1']
        x2 = row['face_coord_x2']
        y1 = row['face_coord_y1']
        y2 = row['face_coord_y2']
        bbox = rectangle(x1, y1, x2, y2)
        cropped_image = get_cropped_image_face(image, bbox)
        feature = get_features_from_image_face(cropped_image, bbox)
        features.append(feature)
    tree = KDTree(features)
    return tree


def save_kdtree(kdtree: KDTree, name_pickle_file: Path):
    """
    Save KDTree.
    Args:
        kdtree:  Built KDTree
        name_pickle_file: Path to save file.
    """
    with open(str(name_pickle_file), 'wb') as file:
        pickle.dump(kdtree, file)


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files", help="Path to csv file or directory with multiple csv files",
                    type=Path)
parser.add_argument("-s", "--save", help="Path to pkl file containing encoded KDTree",
                    type=Path)
parser.add_argument("-b", "--base", help="Image database path",
                    default=Path('face_recognition/data/external/images_actors_and_actresses_full'))
args = parser.parse_args()

kdtree = create_kdtree(path_to_csv=args.files,
                       path_to_images_from_csv_files=args.base)
save_kdtree(kdtree, name_pickle_file=args.save)
