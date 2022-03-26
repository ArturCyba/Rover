import cv2
from matplotlib.pyplot import figure, draw, pause, close
import time
import keyboard
import numpy as np
import math

dist = np.matrix([-0.3612, 0.2793, 0.0007963, -0.0007907, -0.2719])
mtx = np.matrix([[1602.6, 0, 1141.6], [0, 1606, 577.2653], [0, 0, 1]])
ArUcoSize = 47

def getClosestPosition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        arucoFrame = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        distance = None
        for corner in corners:
            rvec1, tvec1, _ = cv2.aruco.estimatePoseSingleMarkers(corner, ArUcoSize, mtx, dist)
            SqEukDist = tvec1[0][0][0] ** 2 + tvec1[0][0][1] ** 2 + tvec1[0][0][2] ** 2
            if distance is not None:
                if SqEukDist < distance:
                    rvec = rvec1
                    tvec = tvec1
            else:
                rvec = rvec1
                tvec = tvec1
        if len(rvec) == 1:
            mat, jac = cv2.Rodrigues(rvec)
        else:
            mat = None
            tvec = None
            rvec = None
    else:
        mat = None
        tvec = None
        rvec = None
    return tvec, mat
