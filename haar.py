import numpy as np
import cv2 as cv
import glob
import pandas as pd

face1 = cv.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
face2 = cv.CascadeClassifier('xml/haarcascade_frontalface_alt_tree.xml')
face3 = cv.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')
face4 = cv.CascadeClassifier('xml/haarcascade_frontalface_alt2.xml')
face5 = cv.CascadeClassifier('xml/haarcascade_profileface.xml')

eye1 = cv.CascadeClassifier('xml/haarcascade_eye.xml')
eye2 = cv.CascadeClassifier('xml/haarcascade_lefteye_2splits.xml')
eye3 = cv.CascadeClassifier('xml/haarcascade_righteye_2splits.xml')

face_cascades = [face1, face2, face3,face4, face5]
eye_cascades = [eye1, eye2, eye3]

"""

UTKFace
all cascades: [1.6457739791073125, 2.3901234567901235, 1.0669671603348359, 1.2066115702479339, 1.2357207615593835]

default eyes + face: [-0.20588793922127255, 0.3012345679012346, -0.4320669671603348, -0.32300275482093666, -0.2901178603807797]

Fairface
all cascades: [3.496, 4.316, 2.997, 2.894, 3.679, 3.3, 3.714]
[3.714, 4.316, 3.19, 3.496, 3.3]

default eyes + face: [0.495, 0.816, 0.307, 0.119, 0.508, 0.395, 0.414]
[0.414, 0.816, 0.3113333333333333, 0.495, 0.395]


50k
basic: [0.416, 0.9047142857142857, 0.355, 0.5348571428571428, 0.3292857142857143]

not basic: [3.6815714285714285, 4.463, 3.2670476190476188, 3.540857142857143, 3.1438571428571427]


"""

def detect_cascade(n, img_name):
    '''
    checks cascades on image and returns the error
    '''
    print(str(n) + ": " + str(img_name))
    image = cv.imread(img_name)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    error = 0
    for i, face_cascade in enumerate(face_cascades):
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        num_faces = len(faces)
        print("faces: " + str(num_faces))
        error += 1 - num_faces
        # if(i == 0):
        #     break

    for i, eye_cascade in enumerate(eye_cascades):
        eyes = eye_cascade.detectMultiScale(gray)
        num_eyes = len(eyes)
        print("eyes: " + str(num_eyes))
        if(i == 0):
            error += 2 - num_eyes
            # break
        else:
            error += 1 - num_eyes

    return error
def search_dataset():
    '''
    Searches dataset and returns pixels caught per race
    '''
    white = glob.glob("/home/hans/uni/critical/critical_studies_week_12/UTKFace/*_*_0_*.jpg")
    black = glob.glob("/home/hans/uni/critical/critical_studies_week_12/UTKFace/*_*_1_*.jpg")
    asian = glob.glob("/home/hans/uni/critical/critical_studies_week_12/UTKFace/*_*_2_*.jpg")
    indian = glob.glob("/home/hans/uni/critical/critical_studies_week_12/UTKFace/*_*_3_*.jpg")
    others = glob.glob("/home/hans/uni/critical/critical_studies_week_12/UTKFace/*_*_4_*.jpg")

    images = [white,black,asian,indian,others]

    errors = [0]*5
    for n, dataset in enumerate(images):
        for img in dataset:
            errors[n] += detect_cascade(n, img)
        errors[n] /= len(images[n])
    print(errors)
    return errors
# search_dataset()


def search_fairface():
    train_df = pd.read_csv('../../fairface/fairface_label_train.csv')
    val_df = pd.read_csv('../../fairface/fairface_label_val.csv')
    df = pd.concat([train_df,val_df], axis=0)

    images = glob.glob("/home/hans/uni/fairface/val/*.jpg")
    train = glob.glob("/home/hans/uni/fairface/train/*.jpg")
    # images.extend(train)

    errors = [0] * 7
    freq = [0] * 7 # To get 1000 of each

    cleanup_nums = {"race": {"Indian": 0, "Black": 1, "Southeast Asian": 2, "East Asian": 3,
    "Middle Eastern": 4, "Latino_Hispanic": 5, "White":6 }}

    train_df = train_df.replace(cleanup_nums)
    for idx, img_name in enumerate(train):
        splitted = img_name.split("/")[5:7]
        lookup = splitted[0]+"/"+splitted[1]
        print(lookup)
        print(idx)
        n = train_df.loc[train_df["file"].str.contains(lookup), "race"].values[0]
        if(freq[n] <= 7000):
            errors[n] += detect_cascade(n, img_name)
            freq[n] += 1
        if(sum(freq) >= 49000):
            break

    for i in range(7):
        errors[i] /= 7000

    print(errors)
    return errors

frequencies = search_fairface()
# frequencies = [0.495, 0.816, 0.307, 0.119, 0.508, 0.395, 0.414]
white = frequencies[6]
asian = (frequencies[2] + frequencies[3] + frequencies[4]) / 3
black = frequencies[1]
indian = frequencies[0]
others = frequencies[5]
frequencies = [white, black, asian, indian, others]
print(frequencies)
