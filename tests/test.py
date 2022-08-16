import cv2
import pytest
from face_recognition.face_recognition import face_landmarks, preprocessing
from pathlib import Path


TESTS_IMAGE_DIR = Path('../tests/test_image')
PATH_TO_ACTORS_AND_ACTRESS = Path('../data/external/images_actors_and_actresses')


def test_single_face_detected(TESTS_IMAGE_1='example_01.jpg'):
    image = cv2.imread(str(TESTS_IMAGE_DIR / TESTS_IMAGE_1))
    assert face_landmarks.get_coords(image) == ([(304, 263, 185, 186)], 1)


def test_face_not_dog_detected(TESTS_IMAGE_2='example_02.jpg'):
    image = cv2.imread(str(TESTS_IMAGE_DIR / TESTS_IMAGE_2))
    assert face_landmarks.get_coords(image) == ([(1895, 830, 958, 959)], 1)


def test_two_face_detected_male_female(TESTS_IMAGE_3='example_03.jpg'):
    image = cv2.imread(str(TESTS_IMAGE_DIR / TESTS_IMAGE_3))
    assert face_landmarks.get_coords(image) == (
        [(2131, 1276, 321, 321), (1703, 1240, 321, 321)], 2)


def test_no_face_detected(TESTS_IMAGE_4='example_04.jpg'):
    image = cv2.imread(str(TESTS_IMAGE_DIR / TESTS_IMAGE_4))
    assert face_landmarks.get_coords(image) == ([], 0)


@pytest.mark.parametrize("image,bbox", preprocessing.list_image_bbox())
def test_face_detector_with_faces_actors_and_actresses(image, bbox):
    name_image = f'{image}.jpg'
    image = cv2.imread(str(PATH_TO_ACTORS_AND_ACTRESS / name_image))
    coordinates, cnt = face_landmarks.get_coords(image)
    assert coordinates == bbox


if __name__ == "__main__":
    pytest.main()
