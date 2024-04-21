import os

folder_path = "C:\\Users\\ThuyLe\\Desktop\\computer_vision_cashew\\data\\vo"
new_prefix = "vo"  # Thay thế bằng tiền tố mới bạn muốn

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
image_files.sort()  # Sắp xếp các tên tập tin theo thứ tự bảng chữ cái

for index, old_name in enumerate(image_files, start=1):
    file_extension = os.path.splitext(old_name)[1]
    new_name = f"{index:03d}_{new_prefix}{file_extension}"  # Sử dụng định dạng số thứ tự có độ dài tối thiểu 3 chữ số (001, 002, ...)
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)
    print(f"Đổi tên '{old_name}' thành '{new_name}'")
