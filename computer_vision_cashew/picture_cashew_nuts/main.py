import os
import re
from skimage.feature import hog
from skimage import io, color
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

parent_folder = 'C:\\Users\\ThuyLe\\Desktop\\computer_vision_cashew\\data\\tong'

def image_processing(image_path, bb_h, bb_w):
    img = cv2.imread(image_path)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.GaussianBlur(grayimg, (5, 5), 0)
    height, width = img.shape[:2]
    # Thu nhỏ ảnh về một nửa kích thước
    new_width = int(width / 3)
    new_height = int(height / 3)
    img = cv2.resize(img, (new_width, new_height))
    edges = cv2.Canny(img, 300, 400)

    # Tìm các contour trên hình ảnh cạnh
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Tìm bounding box có diện tích lớn nhất
    max_area = 0
    max_box = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_box = cv2.boundingRect(cnt)

    if max_box is not None:
        x, y, w, h = max_box
        center_x = x + w // 2
        center_y = y + h // 2
            
        # Tính tọa độ góc trái trên của bounding box mới
        new_x = center_x - (bb_h//2)
        new_y = center_y - (bb_w//2)
            
        #img = cv2.rectangle(img.copy(), (new_x, new_y), (new_x + bb_h, new_y + bb_w), (0, 255, 0), 2)
        roi = grayimg[new_y:new_y+bb_h, new_x:new_x+bb_w]

        if roi.shape[0] >= 8 and roi.shape[1] >= 8:
            roi_hog_feature = hog(roi, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
        
            return roi_hog_feature
    return None



features_all = []
labels = []

for filename in os.listdir(parent_folder):
    if os.path.isfile(os.path.join(parent_folder, filename)):
        file_path = os.path.join(parent_folder, filename)
        if re.search(r'lanh', filename, re.IGNORECASE):
            label = 1  # Gán nhãn 1 là hạt lành
        else:
            label = 0  # Gán nhãn 0 là hạt vỡ
        feature = image_processing(file_path, 120, 120)
        
        features_all.append(feature)
        labels.append(label)

#tạo frame
data = {'Features': features_all, 'Label': labels}
df = pd.DataFrame(data)
# Lưu DataFrame vào tệp CSV
df.to_csv('ds_feature.csv', index=False)

X = np.array(df['Features'].tolist())
Y = np.array(df['Label'].tolist())
print(X.shape)
print(features_all.shape)
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Tạo một mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Số lân cận (k) có thể thay đổi

# Huấn luyện mô hình trên tập huấn luyện
knn_model.fit(x_train, y_train)
knn_accuracy = knn_model.score(x_test, y_test)
print(f"KNN Accuracy: {knn_accuracy}")