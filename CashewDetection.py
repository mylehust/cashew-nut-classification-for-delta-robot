# Program To Read video
# and Extract Frames
import cv2
import numpy as np
import serial
import time
# Function to extract frames

STOP_CONVEYOR_COMMAND_CODE = 10000
SEND_CENTROID_POINTS_COMMAND_CODE = [10001, 10020]
SEND_FLAGS = {'stop_conveyor':1, 'send_centroids':2}

def get_stop_conveyor_command():
    command = str(STOP_CONVEYOR_COMMAND_CODE)+" 0 " + "0"
    return command

def get_send_centroids_commands(centroid_points):
    commands = []
    current_command_code = SEND_CENTROID_POINTS_COMMAND_CODE[0]
    for centroid_point in centroid_points:
        command = str(current_command_code) + " " + str(centroid_point[0]) + " " + str(centroid_point[1])
        commands.append(command)
        current_command_code = current_command_code + 1
    if(current_command_code < SEND_CENTROID_POINTS_COMMAND_CODE[1]):
        for i in range(current_command_code, SEND_CENTROID_POINTS_COMMAND_CODE[1]):
            command = str(i) + " " + str(-500) + " " + str(-500)
            commands.append(command)
    return commands

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

def find_contour_areas(contours):
    areas = []
    for cnt in contours:
        cont_area = cv2.contourArea(cnt)
        areas.append(cont_area)
    return areas

def transform_to_delta_points(x, y):
    x_delta = int(x*0.67 - 150)
    y_delta = int(x*0.67 + 133)
    return x_delta, y_delta

def get_centroid_points(masked_region):
    cv2.imshow("Masked region", masked_region)
    black_img = np.zeros((masked_region.shape[0], masked_region.shape[1], 3))
    gray_img = cv2.cvtColor(masked_region, cv2.COLOR_BGR2YCR_CB)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('black and white', blackAndWhiteImage)
    _, threshold = cv2.threshold(blackAndWhiteImage[:, :, [1]], 200, 450, cv2.THRESH_BINARY)
    cv2.imshow('Cb frame', threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    sorted_contours_area = sorted(contours, key = cv2.contourArea, reverse=True)
    sorted_areas = find_contour_areas(sorted_contours_area)
    max_none_cashew_size = int(sorted_areas[0]*0.7)
    
    result_contour = []
    rectangles = []
    delta_centroid_points = []
    for sc in sorted_contours_area:
        area = cv2.contourArea(sc)
        if (area < max_none_cashew_size) and area > 100:
            result_contour.append(sc)
    for cnt in result_contour:
        (x, y, w, h) = cv2.boundingRect(cnt) 
        rectangles.append((x, y, w, h))
    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        cv2.rectangle(black_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        x_c, y_c = int((2*x + w)/2), int((2*y+h)/2)
        x_delta, y_delta = transform_to_delta_points(x_c, y_c)
        delta_centroid_points.append((x_delta, y_delta))
        cv2.line(black_img, (x_c, y_c), (x_c+1, y_c+1), (255, 0, 0), 5)
    return black_img, delta_centroid_points

if __name__ == '__main__':
    frame = cv2.imread("data/raw_data/frame_330.png")
    cv2.imshow('frame', frame)
    # x_start, x_end, y_start, y_end = 0, 640, 130, 258
    x_start, x_end, y_start, y_end = 0, 640, 210, 338
    grid_height, grid_width = 64, 64
    
    frame = frame[y_start:y_end, x_start:x_end, :]
    grid_frame = draw_grid(image=frame, 
                x_start=0, x_end=frame.shape[1], y_start=0, y_end=frame.shape[0],
                grid_height=grid_height, grid_width=grid_width)
    black_img, centroid_points = get_centroid_points(frame)
    cv2.imshow('black image', black_img)
    cv2.imshow('grid frame', grid_frame)
    print(centroid_points)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    #Sending centroids
    ser = serial.Serial('COM3', 9600, timeout=1)
    ser.reset_input_buffer()
    send_flag = SEND_FLAGS['stop_conveyor']
    while True:
        if send_flag == SEND_FLAGS['stop_conveyor']:
            command = get_stop_conveyor_command()                                                                                                                                                                                                       
            print("[+] Sending: " + command)    
            ser.write(command.encode())
            time.sleep(2)
            send_flag = SEND_FLAGS['send_centroids']
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
        elif send_flag == SEND_FLAGS['send_centroids']:
            commands = get_send_centroids_commands(centroid_points=centroid_points)
            for com in commands:
                command = command + com + " "
            print("[+] Sending: " + command) 
            ser.write(command.encode())
            time.sleep(2)
            send_flag = SEND_FLAGS['stop_conveyor']
            line = ser.readline().decode('utf-8').rstrip()
            print(line)

    ser.close()
    