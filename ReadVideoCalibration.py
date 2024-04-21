# Program To Read video
# and Extract Frames
import cv2
import yaml
import numpy as np
# Function to extract frames
def FrameCapture():
    vidObj = cv2.VideoCapture(0)
    count = 0
    success = 1
    with open('calibration/calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)

    mtx = np.array(loadeddict.get('camera_matrix'))
    dist = np.array(loadeddict.get('dist_coeff'))
    while success:
        success, frame = vidObj.read()
        cv2.imshow('frame', frame)
        # undistort
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('calibresult', dst)
        print(dst.shape)
        print(frame.shape)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Driver Code
if __name__ == '__main__':
    FrameCapture()