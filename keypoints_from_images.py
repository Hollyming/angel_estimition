# keypoints_from_images.py
 
import sys
import cv2
import os
from sys import platform
import argparse
import time
import math
 
 
def angle_between_points( p0, p1, p2 ):
    # 计算角度
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1.0 
    
    return  math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi
 
 
def length_between_points(p0, p1):
    # 2点之间的距离
    return math.hypot(p1[0]- p0[0], p1[1]-p0[1])
 
 
def get_angle_point(human, pos):
    # 返回各个部位的关键点
    pnts = []
 
    if pos == 'left_elbow':
        pos_list = (5,6,7)
    elif pos == 'left_hand':
        pos_list = (1,5,7)
    elif pos == 'left_knee':
        pos_list = (12,13,14)
    elif pos == 'left_ankle':
        pos_list = (5,12,14)
    elif pos == 'right_elbow':
        pos_list = (2,3,4)
    elif pos == 'right_hand':
        pos_list = (1,2,4)
    elif pos == 'right_knee':
        pos_list = (9,10,11)
    elif pos == 'right_ankle':
        pos_list = (2,9,11)
    else:
        print('Unknown  [%s]', pos)
        return pnts
 
    for i in range(3):
        if human[pos_list[i]][2] <= 0.1:
            print('component [%d] incomplete'%(pos_list[i]))
            return pnts
 
        pnts.append((int( human[pos_list[i]][0]), int( human[pos_list[i]][1])))
    return pnts
 
 
def angle_left_hand(human):
    pnts = get_angle_point(human, 'left_hand')
    if len(pnts) != 3:
        print('component incomplete')
        return -1
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left hand angle:%f'%(angle))
    return angle
    
 
def angle_left_elbow(human):
    pnts = get_angle_point(human, 'left_elbow')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left elbow angle:%f'%(angle))
    return angle
    
 
def angle_left_knee(human):
    pnts = get_angle_point(human, 'left_knee')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left knee angle:%f'%(angle))
    return angle
    
 
def angle_left_ankle(human):
    pnts = get_angle_point(human, 'left_ankle')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left ankle angle:%f'%(angle))
    return angle
    
 
def angle_right_hand(human):
    pnts = get_angle_point(human, 'right_hand')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right hand angle:%f'%(angle))
    return angle
 
 
def angle_right_elbow(human):
    pnts = get_angle_point(human, 'right_elbow')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right elbow angle:%f'%(angle))
    return angle
    
 
def angle_right_knee(human):
    pnts = get_angle_point(human, 'right_knee')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right knee angle:%f'%(angle))
    return angle
    
 
def angle_right_ankle(human):
    pnts = get_angle_point(human, 'right_ankle')
    if len(pnts) != 3:
        print('component incomplete')
        return
 
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right ankle angle:%f'%(angle))
    return angle
 
 
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../build/python/openpose/Debug');
            # sys.path.append(dir_path + '/../../build/python/openpose/Debug')
            # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../build/x64/Debug;' +  dir_path + '/../../bin;'
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../build/x64/Debug;' +  dir_path + '/../../build/bin;'
            from openpose import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
 
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="images", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()
 
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../models/"
 
    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item
 
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
 
    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
 
    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
 
        print("Body keypoints: \n" + str(datum.poseKeypoints))
 
        human_count = len(datum.poseKeypoints)
 
        for i in range(human_count):
            for j in range(25):
                print(datum.poseKeypoints[i][j][0])
 
        for i in range(human_count):
            angle_left_hand(datum.poseKeypoints[i] )
            angle_left_elbow(datum.poseKeypoints[i] )
            angle_left_knee(datum.poseKeypoints[i] )
            angle_left_ankle(datum.poseKeypoints[i] )
            angle_right_hand(datum.poseKeypoints[i] )
            angle_right_elbow(datum.poseKeypoints[i] )
            angle_right_knee(datum.poseKeypoints[i] )
            angle_right_ankle(datum.poseKeypoints[i] )
 
        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            cv2.imwrite("{}.jpg".format(time.time()), datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break
 
except Exception as e:
    print(e)
    sys.exit(-1)