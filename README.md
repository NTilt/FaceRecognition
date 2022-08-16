# Face Recognition #

This module contains set of methods to detect faces and extract features from them. 
These features can be used to search for people in the database.

### Setup ###

* Make sure that you have python 3.8
* Install requirements from the `requirements.txt`

### Launch unit-tests
In order to run tests to evaluate the correct operation of the face detector, go to the 
```
cd face_recognition/tests
``` 
folder and write the command to the terminal:
```
pytest test.py
```

## Usage ##

### Command-line Interface ###


There are currently 3 main commands available for use:

- `detecting_face` - Recognition of faces in an image and saving their coordinates to a separate csv-file  
- `create_kdtree` - Builds a KDTree based on one or more csv files and saves it to a file
- `classification_people` - Detection and classification faces in image

#### `detecting_face` command line tool ####

The `detecting_face` command requires one photo to be passed as the first parameter to this function. 
In order to find out the result, it is necessary to pass as the second parameter the path to the csv-file, 
which will store the coordinates of the faces found in the photo. 
This function has a third (optional) parameter that is responsible for saving the input photo 
(by default, the photo will be automatically saved to the folder: `face_recognition/data/external/save_images`)

```
$ detecting_face.py -i test_image.jpeg -s test_csv_file.csv
```
#### `create_kdtree` command line tool ####

The `create_kdtree` command takes as parameters a csv file, or a path to a folder with several csv files, 
on the basis of which the kdtree of features obtained from images will be built, as the second parameter 
it is necessary to pass the path to the file in which the resulting kdtree will be saved.


```
$ create_kdtree.py -f all_csv_files -s test_save_kdtree.pkl
```
или 
```
$ create_kdtree.py -f test.csv -s test_save_kdtree.pkl
```

#### `classification_people` command line tool ####


The `classification_people` command takes two parameters, the first is 
the photo of the person to be identified, the second is the path to the kdtree file
(default: `face_recognition/data/processed/features_index.pkl`)
containing the features to be compared with the new image.
```
$ classification_people.py -i new_image.jpeg -db path_to_kdtree
```