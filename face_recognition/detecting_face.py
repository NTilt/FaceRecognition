import cv2
from pathlib import Path
import pandas as pd
import argparse
from main import get_face_coordinates, get_face_id


def detecting_faces(path_to_image: Path, path_to_save_image: Path):
    """
    Create csv-file with coordinates, some name and path to the original file with faces
    Args:
        path_to_image: Path to the image to be analyzed
        path_to_save_image: Path to the file where the original images are stored
    """
    image = cv2.imread(str(path_to_image))
    name_image = get_face_id(image)
    rects = get_face_coordinates(image)
    cv2.imwrite(str(path_to_save_image / f'{name_image}.jpeg'), image)
    faces = []
    for ind, face_coordinates in enumerate(rects):
        x1 = face_coordinates.left()
        y1 = face_coordinates.top()
        x2 = face_coordinates.right()
        y2 = face_coordinates.bottom()
        faces.append({
            'name': f'face_{ind + 1}',
            'face_coord_x1': x1,
            'face_coord_x2': x2,
            'face_coord_y1': y1,
            'face_coord_y2': y2,
            'path_to_image': (path_to_save_image / f'{name_image}.jpeg')
        })
    df = pd.DataFrame(faces)
    return df


def save_faces(df: pd.DataFrame, path_to_csv_file: Path):
    """
    Saves the resulting DataFrame to a file.
    Args:
        df: pandas DataFrame
        path_to_csv_file: Path to save file
    """
    if path_to_csv_file.is_file():
        df.to_csv(str(path_to_csv_file), mode='a', header=False, index=False)
    else:
        df.to_csv(str(path_to_csv_file), index=False)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to image for detecting faces", type=Path)
parser.add_argument("-s", "--save", help="Path to csv-file with face coordinates on image",
                    type=Path)
parser.add_argument("-b", "--base", help="Image database path",
                    default=Path('face_recognition/data/external/save_images'))
args = parser.parse_args()

data = detecting_faces(path_to_image=args.image,
                       path_to_save_image=args.base)
save_faces(data, path_to_csv_file=args.save)
