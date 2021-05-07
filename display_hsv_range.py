from statistics import mean
import numpy as np
import cv2 as cv
import math
import os
import glob
import pandas as pd

from numba import jit, cuda

'''

fair face dataset size by ethnicity:
29275
36656
36656
36656
36651
36646
36444

===https://nalinc.github.io/blog/2018/skin-detection-python-opencv/

//average frequency of tones per image: [0.7290860588793924, 0.731716975308642, 0.6472363329040566, 0.7370691976584026, 0.7416862647325478]

fair face:
--------
[0.2778647655555602, 0.21334754185998558, 0.1952669842007727, 0.2151597318077645, 0.1598463332944599, 0.2488419881158183, 0.28739257854061534]
[0.28739257854061534, 0.19009101643433235, 0.21334754185998558, 0.2778647655555602, 0.2488419881158183]

fair face (YCrCb):
--------
[0.29583450677185924, 0.2353370821526219, 0.20809051407922313, 0.23459418318927236, 0.17503786904741048, 0.26526544861492674, 0.3234158840674478]
[0.3234158840674478, 0.20590752210530197, 0.2353370821526219, 0.29583450677185924, 0.26526544861492674]

===https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

fair face:
--------
[0.21433046198304267, 0.1501218319546634, 0.16090591856516612, 0.1847618463975816, 0.1322107187855497, 0.2100126095683654, 0.25040623277602664]
[0.25040623277602664, 0.15929282791609914, 0.1501218319546634, 0.21433046198304267, 0.2100126095683654]


===https://github.com/Jeanvit/PySkinDetection

fair face:
--------
[0.3031470382239136, 0.23350997091721332, 0.21058813613357277, 0.23770578980582815, 0.17538597279917292, 0.2695655586392549, 0.3162163854508606]
[0.3162163854508606, 0.20789329957952463, 0.23350997091721332, 0.3031470382239136, 0.2695655586392549]


fair face (YCrCb):
--------
[0.2542776722265543, 0.19846356560393993, 0.18864558343456833, 0.21203924929773202, 0.15449081403659107, 0.23989353143582975, 0.29358483787381234]
[0.29358483787381234, 0.1850585489229638, 0.19846356560393993, 0.2542776722265543, 0.23989353143582975]



'''

light_ranges = [[249,207,189], [218,158,128],[253,193,172],[249,214,202],[251,194,176],[251,178,152]]
dark_ranges = [[37, 150, 190], [102,83,73],[122,91,77],[183,136,114],[183,136,114],[134,108,97]]

skin_ranges = [[45, 34, 30],
[60, 46, 40],
[75, 57, 50],
[90, 69, 60],
[105, 80, 70],
[120, 92, 80],
[135, 103, 90],
[150, 114, 100],
[165, 126, 110],
[180, 138, 120],
[195, 149, 130],
[210, 161, 140],
[225, 172, 150],
[240, 184, 160],
[255, 195, 170],
[255, 206, 180],
[255, 218, 190],
[255, 229, 200]]

# light_ranges = skin_ranges[9:]
# dark_ranges = skin_ranges[:9]


os.system('clear') # fixes buggy cv2 messages
# first article

min_HSV = np.array([0, 58, 30], dtype = "uint8")
max_HSV = np.array([33, 255, 255], dtype = "uint8")

# min_YCrCb = np.array([0,133,77],np.uint8)
# max_YCrCb = np.array([235,173,127],np.uint8)


# second article
# min_HSV = np.array([0, 48, 80], dtype = "uint8")
# max_HSV = np.array([20, 255, 255], dtype = "uint8")

# third article
# min_HSV = np.array([0, 40, 0], dtype = "uint8")
# max_HSV = np.array([25, 255, 255], dtype = "uint8")
#
# min_YCrCb = np.array((0, 138, 67), dtype = "uint8")
# max_YCrCb = np.array((255, 173, 133), dtype = "uint8")

# dir = np.array(max_HSV - min_HSV, dtype = "uint8")

# print(dir)

# gets range from list of rgb values
def get_max_range(rgbs):
    minR = 255
    minG = 255
    minB = 255
    maxR = 0
    maxG = 0
    maxB = 0
    for rgb in rgbs:
        if(rgb[0] < minR):
            minR = rgb[0]
        if(rgb[1] < minG):
            minG = rgb[1]
        if(rgb[2] < minB):
            minB = rgb[2]
        if(rgb[0] > maxR):
            maxR = rgb[0]
        if(rgb[1] > minG):
            maxG = rgb[1]
        if(rgb[2] > minB):
            maxB = rgb[2]
    return [[minR, minG, minB], [maxR, maxG, maxB]]

avg_dark_range = map(mean, zip(*dark_ranges))
avg_light_range = map(mean, zip(*light_ranges))

def inRange(rgb, rgbs):
    for rnge in rgbs:
        if(np.linalg.norm(rnge - rgb) < 1):
            return True
    return False

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

    frequencies = [0]*5


    for n, dataset in enumerate(images):
        for img in dataset:
            frequencies[n] += skin_pixels(n, img)
        frequencies[n] /= len(images[n])
    print(frequencies)
    return frequencies

# @jit(target ="cuda")
def search_fairface():
    '''
    Same as search_dataset for the Fairface dataset
    '''
    train_df = pd.read_csv('../../fairface/fairface_label_train.csv')
    val_df = pd.read_csv('../../fairface/fairface_label_val.csv')
    df = pd.concat([train_df,val_df], axis=0)
    images = glob.glob("/home/hans/uni/fairface/val/*.jpg")
    train = glob.glob("/home/hans/uni/fairface/train/*.jpg")
    # images.extend(train)

    cleanup_nums = {"race": {"Indian": 0, "Black": 1, "Southeast Asian": 2, "East Asian": 3,
    "Middle Eastern": 4, "Latino_Hispanic": 5, "White":6 }}
    train_df = train_df.replace(cleanup_nums)


    image = cv.imread(train[0])
    print(image.shape)

    frequencies = [0]*7
    freq = [0] * 7 # To get 1000 of each


    for img_name in train:


        splitted = img_name.split("/")[5:7]
        lookup = splitted[0]+"/"+splitted[1]
        print(lookup)
        n = train_df.loc[train_df["file"].str.contains(lookup), "race"].values[0]
        # if(freq[n] <= 1000):
        frequencies[n] += skin_pixels(n, img_name)
            # freq[n] += 1
        # if(sum(freq) >= 7000):
        #     break

    for i in range(7):
        frequencies[i] /= (len(train_df.loc[train_df["file"].str.contains(str(i)), "race"].values))

    print(frequencies)
    white = frequencies[6]
    asian = (frequencies[2] + frequencies[3] + frequencies[4]) / 3
    black = frequencies[1]
    indian = frequencies[0]
    others = frequencies[5]
    frequencies = [white, asian, black, indian, others]
    print(frequencies)
    return frequencies

def skin_pixels(n, img_name):
    '''
    returns amount of pixels that are classified as skin in an image
    '''
    print(str(n) + ": " + str(img_name))
    image = cv.imread(img_name)
    imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    skinRegionHSV = cv.inRange(imageHSV, min_HSV, max_HSV)
    skinHSV = cv.bitwise_and(image, image, mask = skinRegionHSV)
    # cv.imwrite("processed2/"+str(n)+"/"+os.path.basename(img_name)+".png", np.hstack([image, skinHSV]))
    # return np.sum(skinRegionHSV) / (255*200*200)
    return np.sum(skinRegionHSV / (255*224*224))

def skin_pixels_ycrcb(n, img_name):
    '''
    returns amount of pixels that are classified as skin in an image
    '''
    print(str(n) + ": " + str(img_name))
    image = cv.imread(img_name)
    imageYCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    skinYCrCb = cv.bitwise_and(image, image, mask = skinRegionYCrCb)
    # cv.imwrite("processed3/"+str(n)+"/"+os.path.basename(img_name)+".png", np.hstack([image, skinYCrCb]))
    # return np.sum(skinRegionHSV) / (255*200*200)

    return np.sum(skinRegionYCrCb / (255*224*224))


def check_tones():
    '''
    Checks what tones the given hsv range picks up on.
    '''

    for n, rgb in enumerate(skin_ranges):
        # Create a blank 300x300 black image
        image = np.zeros((20, 1, 3), np.uint8)
        # Fill image with red color(set each pixel to red)
        for i in range(10):
            image[i] = [x-i for x in rgb]
            image[i+10] = [x+i for x in rgb]
        hsv = cv.cvtColor(image,cv.COLOR_RGB2HSV)
        skinRegionHSV = cv.inRange(hsv, min_HSV, max_HSV)
        # cv.imshow('ROI',cv.cvtColor(skinRegionHSV,cv.COLOR_HSV2BGR))
        # cv.waitKey(0)
        print(rgb)
        print(skinRegionHSV)

def display():

    light_hits = 0
    dark_hits = 0

    s_gradient = np.ones((500,1), dtype=np.uint8)*np.linspace(min_HSV[1], max_HSV[1], 500, dtype=np.uint8)
    v_gradient = np.rot90(np.ones((500,1), dtype=np.uint8)*np.linspace(min_HSV[2], max_HSV[2], 500, dtype=np.uint8))
    h_array = np.arange(min_HSV[0], max_HSV[0]+1)

    for n, hue in enumerate(h_array):
        print(n)
        h = hue*np.ones((500,500), dtype=np.uint8)
        hsv_colour = cv.merge((h, s_gradient, v_gradient))
        rgb_colour = cv.cvtColor(hsv_colour, cv.COLOR_HSV2BGR)
        rgb = cv.cvtColor(rgb_colour, cv.COLOR_BGR2RGB)

        # for x in range(500):
        #     for y in range(500):
        #         if(inRange(rgb[x][y], light_ranges)):
        #             # print("found light skin tone " + str(n))
        #             light_hits+=1
        #         if(inRange(rgb[x][y], dark_ranges)):
        #             # print("found dark skin tone " + str(n))
        #             dark_hits+=1

        cv.imshow('', rgb_colour)
        cv.imwrite("img/_hsv_" + str(n)+".png", rgb_colour)
        cv.waitKey(250)


    cv.destroyAllWindows()
    return light_hits, dark_hits

# light_hits, dark_hits = display()
# print("light hits: " + str(light_hits))
# print("dark hits: " + str(dark_hits))
# check_tones()
# search_dataset()
search_fairface()
