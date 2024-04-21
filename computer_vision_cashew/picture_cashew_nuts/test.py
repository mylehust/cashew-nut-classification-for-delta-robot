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
import joblib
parent_folder = 'C:\\Users\\ThuyLe\\Desktop\\computer_vision_cashew\\data\\hihi'

def image_processing(image_path, bb_h, bb_w):
    img = cv2.imread(image_path)
    
    #img = cv2.GaussianBlur(grayimg, (5, 5), 0)
    height, width = img.shape[:2]
    # Thu nhỏ ảnh về một nửa kích thước
    new_width = int(width / 3)
    new_height = int(height / 3)
    img = cv2.resize(img, (new_width, new_height))
    edges = cv2.Canny(img, 300, 400)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        roi = cv2.blur(roi, (5, 5))
        #cv2.imshow('roi', roi)
        if roi.shape[0] >= 8 and roi.shape[1] >= 8:
            roi_hog_feature = hog(roi, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
        
            return roi_hog_feature, new_x, new_y
    return None, None, None


bb_h, bb_w = 110, 110
features_all = []
labels = []
feature_hog = []
feature_area = []
feature_color = []

def are(img_path, new_x, new_y):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = gray_image[new_y:new_y+bb_h, new_x:new_x+bb_w]

    # Sử dụng hàm threshold để tạo hình ảnh nhị phân (binary image)
    _, binary_image = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tìm các contours trong hình ảnh nhị phân
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo danh sách để lưu trữ các diện tích của các hạt điều
    areas = []

    # Duyệt qua từng contour và tính toán diện tích
    for contour in contours:
        area = cv2.contourArea(contour)  # Tính toán diện tích
        areas.append(area)

    # Tính trung bình diện tích của các hạt điều
    average_area = np.mean(areas)
    return average_area

def color(img_path, new_x, new_y):
    image = cv2.imread(img_path)
    roi = image[new_y:new_y+bb_h, new_x:new_x+bb_w] #ảnh trong vùng bounding box
    hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Trích xuất kênh màu Hue (màu sắc)
    hue_channel = hsv_image[:, :, 0]
    # Tính toán các thống kê hoặc đặc trưng từ kênh Hue (ví dụ: trung bình màu sắc)
    average_hue = np.mean(hue_channel)
    return average_hue

for filename in os.listdir(parent_folder):
    if os.path.isfile(os.path.join(parent_folder, filename)):
        file_path = os.path.join(parent_folder, filename)
        if re.search(r'lanh', filename, re.IGNORECASE):
            label = 1  # Gán nhãn 1 là hạt lành
        else:
            label = 0  # Gán nhãn 0 là hạt vỡ
        feature, new_x, new_y = image_processing(file_path, bb_h, bb_w)
        average_area = are(file_path, new_x, new_y)
        average_hue = color(file_path, new_x, new_y)
        feature_hog.append(feature)
        feature_area.append(average_area)
        feature_color.append(average_hue)
        labels.append(label)


# Chuyển danh sách feature_area và feature_hog thành mảng numpy
feature_area = np.array(feature_area)
feature_hog = np.array(feature_hog)
feature_color = np.array(feature_color)
#print(feature_hog.shape)
# # Kết hợp hai mảng numpy thành một mảng duy nhất
#features_all = np.column_stack((feature_area, feature_hog))
features_all = feature_hog
 #tạo frame
data = {'Features': features_all.tolist(), 'Label': labels}
df = pd.DataFrame(data)
# # Lưu DataFrame vào tệp CSV
df.to_csv('ds_feature.csv', index=False)

X = features_all  # Đặc trưng
Y = labels  # Nhãn

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Tạo một mô hình KNN

# Sử dụng khoảng cách Euclidean (mặc định)
count = 7
knn_euclidean = KNeighborsClassifier(n_neighbors=count)
knn_euclidean.fit(x_train, y_train)
y_pred_euclidean = knn_euclidean.predict(x_test)
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)
print(f'Độ chính xác (Euclidean): {accuracy_euclidean * 100:.2f}%')

# Sử dụng khoảng cách Manhattan (L1)
knn_manhattan = KNeighborsClassifier(n_neighbors=count, metric='manhattan')
knn_manhattan.fit(x_train, y_train)
y_pred_manhattan = knn_manhattan.predict(x_test)
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)
print(f'Độ chính xác (Manhattan): {accuracy_manhattan * 100:.2f}%')

# Sử dụng khoảng cách Cosine Similarity
knn_cosine = KNeighborsClassifier(n_neighbors=count, metric='cosine')
knn_cosine.fit(x_train, y_train)
y_pred_cosine = knn_cosine.predict(x_test)
accuracy_cosine = accuracy_score(y_test, y_pred_cosine)
print(f'Độ chính xác (Cosine): {accuracy_cosine * 100:.2f}%')

# Sử dụng khoảng cách Chebyshev
knn_chebyshev = KNeighborsClassifier(n_neighbors=count, metric='chebyshev')
knn_chebyshev.fit(x_train, y_train)
y_pred_chebyshev = knn_chebyshev.predict(x_test)
accuracy_chebyshev = accuracy_score(y_test, y_pred_chebyshev)
print(f'Độ chính xác (Chebyshev): {accuracy_chebyshev * 100:.2f}%')

joblib.dump(knn_manhattan, 'Model_cashew.pkl')


