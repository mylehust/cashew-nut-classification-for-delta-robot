import cv2
import numpy as np
import yaml

class FrameCalibration:
    def __init__(self)->None:
        pass
        
    def get_calibrate_parameter(self, frame, path_to_calib_file):
        with open(path_to_calib_file) as f:
            loadeddict = yaml.load(f, Loader=yaml.FullLoader)
        mtx = np.array(loadeddict.get('camera_matrix'))
        dist = np.array(loadeddict.get('dist_coeff'))
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        return mtx, dist, newcameramtx, roi
    
    def calibrate_frame(self, frame, mtx, dist, newcameramtx, roi):
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst