import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests

# Đọc ảnh grayscale
img = cv2.imread('C:\\Users\\ThuyLe\\Desktop\\xla\\xla1\\xla_thuchanh\\hand_writing.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # màu RGB
img_gauss = cv2.GaussianBlur(gray_img, (3, 3), 0)
_, img_binary = cv2.threshold(img_gauss, 150, 255, cv2.THRESH_BINARY) # ảnh nhị phân

# Tính gradient bằng Sobel
gradient_x = cv2.Sobel(img_binary, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img_binary, cv2.CV_64F, 0, 1, ksize=3)

# Tính gradient tổng hợp
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Tính độ lệch chuẩn của gradient
standard_deviation = np.std(gradient_magnitude)

# Xác định ngưỡng cao và thấp
high_threshold = 2 * standard_deviation
low_threshold = high_threshold / 2
print("High Threshold:", high_threshold)
print("Low Threshold:", low_threshold)

# Sử dụng hàm Canny để tạo ảnh biên
imgCanny = cv2.Canny(img_binary, low_threshold, high_threshold)

# xác định Contour
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Vì drawContours sẽ thay đổi ảnh gốc nên cần lưu ảnh sang một biến mới.
imgOrigin = gray_img.copy() # ảnh gốc
img1 = img_binary.copy() # ảnh nhị phan

# Vẽ toàn bộ contours trên hình ảnh gốc
cv2.drawContours(img1, contours, -1 , (0, 255, 0), 3)

# Tìm ra diện tích của toàn bộ các contours
area_cnt = [cv2.contourArea(cnt) for cnt in contours]
area_sort = np.argsort(area_cnt)[::-1]

# Vẽ bounding box cho contours có diện tích lớn thứ 1
cnt = contours[area_sort[0]]
x,y,w,h = cv2.boundingRect(cnt)
print('centroid: ({}, {}), (width, height): ({}, {})'.format(x, y, w, h))
img_bb = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('a',img_bb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Vẽ contour
# plt.subplot(131),plt.imshow(imgOrigin),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(imgCanny),plt.title('Canny Binary Image')
# plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img1),plt.title('All Contours')
# plt.xticks([]), plt.yticks([])
# plt.show()


# import cv2
# import numpy as np

# # Đọc ảnh grayscale
# img = cv2.imread('C:\\Users\\ThuyLe\\Desktop\\computer_vision_cashew\\data\\tong\\020_vo.jpg')
# #img = cv2.imread('picture_cashew_nuts\\data\\raw_data\\frame_200.png')

# # Lấy kích thước ban đầu của ảnh
# height, width = img.shape[:2]

# # Thu nhỏ ảnh về một nửa kích thước
# new_width = int(width / 3)
# new_height = int(height / 3)
# img = cv2.resize(img, (new_width, new_height))

# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ảnh xám
# img_gauss = cv2.GaussianBlur(gray_img, (3, 3), 0)
# _, img_binary = cv2.threshold(img_gauss, 150, 255, cv2.THRESH_BINARY) # ảnh nhị phân

# # Tính gradient bằng Sobel
# gradient_x = cv2.Sobel(img_binary, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(img_binary, cv2.CV_64F, 0, 1, ksize=3)

# # Tính gradient tổng hợp
# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# # Tính độ lệch chuẩn của gradient
# standard_deviation = np.std(gradient_magnitude)

# # Xác định ngưỡng cao và thấp
# high_threshold = 2 * standard_deviation
# low_threshold = high_threshold / 2

# # Sử dụng hàm Canny để tạo ảnh biên
# imgCanny = cv2.Canny(img_binary, low_threshold, high_threshold) #Xác định Contour
# contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #Bounding box cho tất cả vật thể có diện tích > 10
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > 50:
#         x, y, w, h = cv2.boundingRect(cnt)
#         img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

# # cut các bounding box 
# # processed_rectangles = []  # Danh sách các bounding box đã xử lý

# # for idx, cnt in enumerate(contours):
# #     area = cv2.contourArea(cnt)
# #     if area > 10:
# #         x, y, w, h = cv2.boundingRect(cnt)
        
# #         is_overlapping = False
# #         for rect in processed_rectangles:
# #             if (x >= rect[0] and y >= rect[1] and x + w <= rect[0] + rect[2] and y + h <= rect[1] + rect[3]):
# #                 is_overlapping = True
# #                 break
        
# #         if not is_overlapping:
# #             processed_rectangles.append((x, y, w, h))
            
# #             cropped_img = img[y:y+h, x:x+w]
# #             cv2.imwrite(f'cut_{idx}.png', cropped_img)


# # Hiển thị ảnh với bounding box
# cv2.imshow('Image with Bounding Boxes', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # #---------------------------------------DÙNG HÀM ĐỂ XÓA CÁC TRÙNG LẶP BOUNDING BOX------------------------------------------------------------------------------


# # def non_max_suppression(boxes, overlapThresh):
# #      # Nếu không có bounding boxes thì trả về empty list
# #   if len(boxes)==0:
# #     return []
# #   # Nếu bounding boxes nguyên thì chuyển sang float.
# #   if boxes.dtype.kind == "i":
# #     boxes = boxes.astype("float")

# #   # Khởi tạo list của index được lựa chọn
# #   pick = []

# #   # Lấy ra tọa độ của các bounding boxes
# #   x1 = boxes[:,0]
# #   y1 = boxes[:,1]
# #   x2 = boxes[:,2]
# #   y2 = boxes[:,3]

# #   # Tính toàn diện tích của các bounding boxes và sắp xếp chúng theo thứ tự từ bottom-right, chính là tọa độ theo y của bounding box
# #   area = (x2 - x1 + 1) * (y2 - y1 + 1)
# #   idxs = np.argsort(y2)
# #   # Khởi tạo một vòng while loop qua các index xuất hiện trong indexes
# #   while len(idxs) > 0:
# #     # Lấy ra index cuối cùng của list các indexes và thêm giá trị index vào danh sách các indexes được lựa chọn
# #     last = len(idxs) - 1
# #     i = idxs[last]
# #     pick.append(i)

# #     # Tìm cặp tọa độ lớn nhất (x, y) là điểm bắt đầu của bounding box và tọa độ nhỏ nhất (x, y) là điểm kết thúc của bounding box
# #     xx1 = np.maximum(x1[i], x1[idxs[:last]])
# #     yy1 = np.maximum(y1[i], y1[idxs[:last]])
# #     xx2 = np.minimum(x2[i], x2[idxs[:last]])
# #     yy2 = np.minimum(y2[i], y2[idxs[:last]])

# #     # Tính toán width và height của bounding box
# #     w = np.maximum(0, xx2 - xx1 + 1)
# #     h = np.maximum(0, yy2 - yy1 + 1)

# #     # Tính toán tỷ lệ diện tích overlap
# #     overlap = (w * h) / area[idxs[:last]]

# #     # Xóa index cuối cùng và index của bounding box mà tỷ lệ diện tích overlap > overlapThreshold
# #     idxs = np.delete(idxs, np.concatenate(([last],
# #       np.where(overlap > overlapThresh)[0])))
# #   # Trả ra list các index được lựa chọn
# #   return boxes[pick].astype("int")
 
# # boundingBoxes = [cv2.boundingRect(cnt) for cnt in contours]
# # boundingBoxes = np.array([(x,y,x+w,y+h) for (x,y,w,h) in boundingBoxes])
# # pick = non_max_suppression(boundingBoxes, 0.5)
# # # Áp dụng non-maximum suppression để loại bỏ các bounding box chồng chéo
# # pick = non_max_suppression(boundingBoxes, 0.5)

# # # Gán nhãn cho các hạt điều và hạt điều vỡ dựa trên màu sắc
# # imgOrigin = img.copy()

# # for idx, (startX, startY, endX, endY) in enumerate(pick):
# #     roi = img[startY:endY, startX:endX]
    
# #     # Tính giá trị trung bình màu sắc trong ROI
# #     mean_color = np.mean(roi, axis=(0, 1))
    
# #     if mean_color[0] > 100:  # Gán nhãn hạt điều vỡ
# #         label = "Cashew Shell"
# #     else:  # Gán nhãn hạt điều lành
# #         label = "Cashew Nut"
# #      # Lưu ảnh con vào tệp với tên nhãn tương ứng
# #     cv2.imwrite(f'{label}_{idx}.png', roi)
# #     cv2.putText(imgOrigin, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
# #     imgOrigin = cv2.rectangle(imgOrigin, (startX, startY), (endX, endY), (0, 255, 0), 2)
# # resized_img = cv2.resize(imgOrigin, (imgOrigin.shape[1] * 2, imgOrigin.shape[0] * 2))


# # # Hiển thị ảnh với nhãn và bounding box
# # cv2.imshow('Image with Labels and Bounding Boxes', resized_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # # # Lặp qua danh sách các contours và cắt và lưu ảnh con dựa trên bounding box
# # # boundingBoxes = [cv2.boundingRect(cnt) for cnt in contours]
# # # boundingBoxes = np.array([(x, y, x+w, y+h) for (x, y, w, h) in boundingBoxes])

# # # # Áp dụng non-maximum suppression để loại bỏ các bounding box chồng chéo
# # # pick = non_max_suppression(boundingBoxes, 0.5)

# # # # Vẽ bounding box cho tất cả các vật thể được xác định
# # # imgOrigin = img.copy()

# # # for (startX, startY, endX, endY) in pick:
# # #     # Lấy ra index của bounding box trong danh sách boundingBoxes gốc
# # #     idx = np.where((boundingBoxes[:, 0] == startX) & (boundingBoxes[:, 1] == startY))[0][0]
    
# # #     # Lấy contour tương ứng
# # #     cnt = contours[idx]
    
# # #     # Tính diện tích của contour
# # #     area = cv2.contourArea(cnt)
    
# # #     # Nếu diện tích lớn hơn 10, thì mới vẽ bounding box
# # #     if area > 10:
# # #         imgOrigin = cv2.rectangle(imgOrigin, (startX, startY), (endX, endY), (0, 255, 0), 2)

# # # # Hiển thị ảnh với bounding box đã vẽ
# # # cv2.imshow('Image with Bounding Boxes', imgOrigin)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
