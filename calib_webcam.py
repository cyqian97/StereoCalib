import numpy as np
import cv2 as cv
import glob


def calib(images,grid_length,grid_width,square_size_mm):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    flags = cv.CALIB_CB_ADAPTIVE_THRESH
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_length*grid_width, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:grid_length, 0:grid_width]
                * square_size_mm).T.reshape(-1, 2)
    objp = np.expand_dims(np.asarray(objp), -2) 
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    num_succ = 0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(
            gray, (grid_length, grid_width), flags)
        # If found, add object points, image points (after refining them)
        print(fname+": ",ret)
        if ret == True:
            num_succ+=1
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(
                gray, corners, (grid_width, grid_width), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            # cv.drawChessboardCorners(img, (grid_length,grid_width), corners2, ret)
            # cv.imshow('img', img)
            # cv.waitKey(500)
    # cv.destroyAllWindows()
    print(f"Checker board detection in {num_succ} out of {images.__len__()} images")
    ret, K, D, rvecs, tvecs, std_intrinsic, std_extrinsic, error_per_view = cv.calibrateCameraExtended(
        objpoints, imgpoints, gray.shape[::-1], None, None, None, None, None, None, None, None, criteria)
    print("K:\n", K)
    print("std K:\n",np.array(std_intrinsic[:4]).T)
    print("D:\n", D)
    print("std D:\n",np.array(std_intrinsic[4:9]).T)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("reproj error: {}".format(mean_error/len(objpoints)))
    return K,D,imgpoints,objpoints



# grid_length = 20
# grid_width = 11
# square_size_mm = 15
# images = glob.glob('Data/*_cam.png')
# calib(images,grid_length,grid_width,square_size_mm)