import os, sys
sys.path.insert(1, '/mnt/c/Users/ts1454/Projects/Jacobs/Code/Python')
from nav_file import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json


def extract_features(img, feature_type ='SIFT'):
    # uses opencv to find matching features between images. 
    # ORB features
    print('Finding {} features'.format(feature_type))
    if feature_type == 'ORB':
        # find the keypoints with ORB
        orb = cv2.ORB_create()
        kp1 = orb.detect(img1,None)
        kp1, des1 = orb.compute(img1, kp1)

    elif feature_type == 'SIFT':
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img,None)

    return kp, des


def match_features(kp_query, des_query, kp_train, des_train, img_query, img_train, img_query_idx, img_train_idx, correspondences, feature_type='SIFT', show=False):
    # the keypoints are now matched. The qeury descriptors are matched against the train descriptors.
    print('matching features')
    if feature_type == 'ORB':       
        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(des_train,des_query)
        matches = sorted(matches, key = lambda x:x.distance)

        good = matches[:50]

    elif feature_type == 'SIFT':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_train,des_query,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        pts = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
                try:
                    pts = {
                        img_query_idx :  kp_train[m.queryIdx].pt,
                        img_train_idx :  kp_query[m.trainIdx].pt,
                    }
                    correspondences[str(m.queryIdx)] = pts
                    #draw_keyPoints(img_train, kp_train[m.queryIdx])
                    #draw_keyPoints(img_query, kp_query[m.trainIdx])
                except IndexError:
                    print('List index error: ', m.trainIdx, m.queryIdx)
    print('found {} matches between {} and {}'.format(len(correspondences.keys()), img_query_idx, img_train_idx))
    # show if set
    if show:
        img3 = cv2.drawMatches(img_train, kp_train, img_query, kp_query, good, None, flags=2)
        plt.imshow(img3),plt.show()

    return correspondences


def draw_keyPoints(img, keypoint, color = (0, 255, 255)):
    # my own function
    x, y = keypoint.pt
    print(x,y)
    img = cv2.circle(img, (int(x), int(y)), 10, color)
    plt.imshow(img)    
    


def create_initial_camera_poses(nav_file, PCTIMES):
    # initialise json and get first image, which will be at the origin of the new world frame
    poses = defaultdict(list)
    nav_file_init = nav_file[nav_file.PCTIME == PCTIMES[0]]
    # calculate relative poses. 
    for i, PCTIME in enumerate(PCTIMES):
        nav_img = nav_file[nav_file.PCTIME == PCTIME]
        pose = {
            'x' : float(nav_img.XCOORD) - float(nav_file_init.XCOORD),
            'y' : float(nav_img.YCOORD) - float(nav_file_init.YCOORD),
            'z' : 2.35,
            'heading' : float(nav_img.HEADING) - float(nav_file_init.HEADING),
        }
        poses['img_{}'.format(i)] = pose

    return poses


def main():

    PCTIMES=[8042,8043]
    M69 = NavFile('M69', PCDATES=1138, PCTIMES=PCTIMES)
    # load two images
    img0 = M69.load_image(1138, 8042, format='opencv', channels='G')
    img1 = M69.load_image(1138, 8043, format='opencv', channels='G')

    # scaling image to a smaller version so feature matching is easier. 
    scale_percent = 25 # percent of original size
    width = int(img0.shape[1] * scale_percent / 100)
    height = int(img0.shape[0] * scale_percent / 100)
    dim = (width, height)
    img0 = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)
    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

    # extract features in the images
    kp0, des0 = extract_features(img0, feature_type='SIFT')
    kp1, des1 = extract_features(img1, feature_type='SIFT')
    # match those features and add to json
    correspondences = defaultdict(list) # matches of the same pixel will be storred here in the form [pt][pts]
    correspondences = match_features(kp0, des0, kp1, des1, img0, img1, 'img_0', 'img_1', correspondences)
    # compute relative poses of the images. 
    nav_file = M69.nav_file    
    poses = create_initial_camera_poses(nav_file, PCTIMES)
    # save poses and tracjed points. 

    with open('data/correspondences.json', 'w') as outfile:
        json.dump(correspondences, outfile)
    
    with open('data/poses.json', 'w') as outfile:
        json.dump(poses, outfile)


if __name__ == "__main__":

    main()