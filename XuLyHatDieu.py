import cv2
import numpy as np
import yaml
import os

def getCbCalibration():
    with open('calibration/calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    mtx = np.array(loadeddict.get('camera_matrix'))
    dist = np.array(loadeddict.get('dist_coeff'))
    return mtx, dist

def draw_grid(image, x_start, x_end, y_start, y_end, grid_width, grid_height):
    grid_image = image.copy()
    crop_img_width = grid_width
    crop_img_height = grid_height
    color = (0, 255, 0)
    thickness = 1
    cols = int((x_end - x_start)/crop_img_width)
    rows = int((y_end - y_start)/crop_img_height)
    #rectangle
    start_point = (x_start, y_start)
    end_point = (x_end, y_end)
    grid_image = cv2.rectangle(grid_image, start_point, end_point, color, thickness)
    #horizontal
    for idx in range(rows):
        start_point = (x_start, y_start + crop_img_height*idx)
        end_point = (x_end, y_start + crop_img_height*idx)
        grid_image = cv2.line(grid_image, start_point, end_point, color, thickness)
    # #vertical
    for idy in range(cols):
        start_point = (x_start + idy*crop_img_width, y_start)
        end_point = (x_start + idy*crop_img_width, y_end)
        grid_image = cv2.line(grid_image, start_point, end_point, color, thickness)
    return grid_image

def crop_and_save_picture(image, x_start, x_end, y_start, y_end, grid_width, grid_height, index):
    img = image[y_start:y_end, x_start:x_end]
    height, width, _ = img.shape
    cols = int(width/grid_width)
    rows = int(height/grid_height)
    crop_images_idx = 0
    for i in range(rows):
        for j in range(cols):
            # Tính toạ độ của ảnh nhỏ
            y1 = i * grid_height
            y2 = (i + 1) * grid_height
            x1 = j * grid_width
            x2 = (j + 1) * grid_width
            # Cắt vật thể và lưu vào ảnh nhỏ
            small_img = img[y1:y2, x1:x2]
            path = 'D:/Code/MachineLearning/HatDieu/data/all_crop_images/crop_images11/'
            cv2.imwrite(path+"crop_image_" + str(i) + "_" + str(j)
                        + "_" +str(index)+ ".png", small_img)
            crop_images_idx += 1
    print("[+] Saving: " + str(cols*rows) + " images with index: " + str(index))
    return 

def getCalibParams():
    with open('calibration/calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)

    mtx = np.array(loadeddict.get('camera_matrix'))
    dist = np.array(loadeddict.get('dist_coeff'))
    return mtx, dist    


def FrameCapture():
    vidObj = cv2.VideoCapture(0)
    vidObj.set(cv2.CAP_PROP_FPS, 10)  # Đặt số khung hình trên giây
    if vidObj is None or not vidObj.isOpened():
        print('Warning: unable to open video source: ', vidObj)
        return
    #video variables
    frame_count = 1
    interval = 5
    success = 1
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))
        
    image_count = 0
    while success:
        success, frame = vidObj.read()
        # #frame_count = frame_count + 1
        # #if frame_count % (interval*fps) == 0:
        # mtx, dist = getCalibParams()
        # # undistort
        # h,  w = frame.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # # crop the image
        # x, y, w, h = roi
        # frame = dst[y:y+h, x:x+w]
        #print(frame.shape)
        
        #proccess area
        x_start, x_end, y_start, y_end = 0, 640, 210, 338
        grid_height, grid_width = 64, 64
        
        grid_frame = draw_grid(image=frame, 
                x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end,
                grid_height=grid_height, grid_width=grid_width)
        
        cv2.imshow('frame', grid_frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            crop_and_save_picture(image=frame, 
                x_start=x_start, x_end=x_end, y_start=y_start, y_end=y_end,
                grid_height=grid_height, grid_width=grid_width, index=image_count)
            image_count = image_count+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return
if __name__ == '__main__':
    FrameCapture()