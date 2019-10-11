# import the necessary packages
from model.augment import augment_faces
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import imthread

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--augment", required=False, default=10,
    help="number of augmented images to generate for training")
args = vars(ap.parse_args())

# generating augmented images for training
# from single image per person
if not args["augment"] == 'False':
    augment_faces(dataset_dir='dataset', amount=int(args["augment"]))

# grab the paths to the input images in our dataset
imagePaths = list(paths.list_images('dataset'))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

def read_faces(data):
    i, imagePath = data[0], data[1]
    print(">> processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model='cnn')

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

multi_threading = imthread.multi_threading(read_faces, max_threads=30)

# loo-p over the image paths
train_images = [(i, imagePath) for i, imagePath in enumerate(imagePaths)]

multi_threading.start(train_images)

# dump the facial encodings + names to disk
print(">> serializing encodings")
data = {"encodings": knownEncodings, "names": knownNames}
f = open('model/encodings.pkl', "wb")
f.write(pickle.dumps(data))
f.close()
