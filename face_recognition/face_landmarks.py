import numpy as np
from imutils import face_utils
import dlib
import cv2
from typing import List, Tuple, Any
import matplotlib.colors as pltc
from random import sample


def plot_image(img: np.ndarray, arr: [int], signature: List[str], thickness: int = 2, fontScale: float = 1.2):
    """
    Draws the bounding box of the face in the image and wait for any key press
    Args:
        img: Original image
        arr: Bounding box coordinates
        thickness: Bounding box thickness (default 2)
        fontScale: Font scale factor that is multiplied by the font-specific base size
        signature: Text at each bounding box

    """

    image = img
    all_colors = [k for k, v in pltc.cnames.items()]
    #colors = sample(all_colors, len(arr))
    colors = [(255, 0, 255), (255, 255, 255)]
    print(arr)
    for i, item in enumerate(arr):
        (x, y, w, h) = item
        cv2.rectangle(image, (x, y), (w, h), color=colors[i], thickness=thickness)
        cv2.putText(image, signature[i].format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale,
                    color=colors[i], thickness=thickness)
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def get_coords(img: np.ndarray) -> List[Tuple[Any, ...]]:
    """
    Returns the coordinates of the bounding box and the number of faces in the image
    Args:
        img: Original image

    Returns:
        List [int]: coordinates of the bounding box
        int: number of faces in the image

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
    arr = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        arr.append((x, y, x + w, y + h))
    return arr



