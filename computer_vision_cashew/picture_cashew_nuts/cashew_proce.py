import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt

img = cv2.imread('C:\\Users\\ThuyLe\\Desktop\\computer_vision_cashew\\data\\shell\\012_shell.jpg')
#img = cv2.imread('picture_cashew_nuts\\data\\raw_data\\frame_200.png')
# Lấy kích thước ban đầu của ảnh
height, width = img.shape[:2]

# Thu nhỏ ảnh về một nửa kích thước
new_width = int(width / 3)
new_height = int(height / 3)
img = cv2.resize(img, (new_width, new_height))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray, 300, 400)

plt.imshow(edges)
plt.show()


# # Find contour
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # Tìm ra diện tích của toàn bộ các contours
# area_cnt = [cv2.contourArea(cnt) for cnt in contours]
# area_sort = np.argsort(area_cnt)[::-1]
# # Trích xuất contour lớn nhất
# cnt = contours[area_sort[0]]
# x,y,w,h = cv2.boundingRect(cnt)
# print('centroid: ({}, {}), (width, height): ({}, {})'.format(x, y, w, h))
# # Vẽ bounding box
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# plt.imshow(img)
# plt.show()
# # Tìm các contour trên hình ảnh cạnh
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Vẽ bounding box cho mỗi contour
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area > 20:
#         x, y, w, h = cv2.boundingRect(cnt)
#         img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Hiển thị hình ảnh với bounding box
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Image with Bounding Boxes')
# plt.show()
