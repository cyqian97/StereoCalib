import numpy as np
import cv2 as cv
import glob
from calib_webcam import calib


grid_length = 20
grid_width = 11
square_size_mm = 15
images1 = glob.glob('Data/*_cam.png')
K1,D1,imgpoints1,objpoints = calib(images1,grid_length,grid_width,square_size_mm)

images2 = glob.glob('Data/*_web.png')
K2,D2,imgpoints2,objpoints = calib(images2,grid_length,grid_width,square_size_mm)

flags = cv.CALIB_FIX_INTRINSIC
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(objpoints,imgpoints1,imgpoints2,K1,D1,K2,D2,None,flags=flags,criteria=criteria)

print(R)
print(T)