import os
import random
import shutil

# 配置路径
src_images_dir = "/home/admin/atec_project/YOLO_door_detect/datasets/images/caiji"
src_labels_dir = "/home/admin/atec_project/YOLO_door_detect/datasets/labels/caiji"

dst_images_train = "/home/admin/atec_project/YOLO_door_detect/datasets/images/train1"
dst_images_val = "/home/admin/atec_project/YOLO_door_detect/datasets/images/val1"
dst_labels_train = "/home/admin/atec_project/YOLO_door_detect/datasets/labels/train1"
dst_labels_val = "/home/admin/atec_project/YOLO_door_detect/datasets/labels/val1"

# 创建目标目录
os.makedirs(dst_images_train, exist_ok=True)
os.makedirs(dst_images_val, exist_ok=True)
os.makedirs(dst_labels_train, exist_ok=True)
os.makedirs(dst_labels_val, exist_ok=True)

# 获取所有图像文件列表（仅支持jpg/jpeg/png格式）
image_files = []
for f in os.listdir(src_images_dir):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(src_images_dir, f)
        # 检查对应的标签文件是否存在
        label_name = os.path.splitext(f)[0] + '.txt'
        label_path = os.path.join(src_labels_dir, label_name)
        if os.path.exists(label_path):
            image_files.append(f)
        else:
            print(f"警告：图像 {f} 没有对应的标签文件，已跳过")

# 打乱文件顺序
random.seed(42)  # 固定随机种子保证可重复性
random.shuffle(image_files)

# 按4:1比例划分
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_files(file_list, img_dst, lbl_dst):
    """复制图像和标签到目标目录"""
    for f in file_list:
        # 复制图像
        src_img = os.path.join(src_images_dir, f)
        shutil.copy2(src_img, img_dst)
        
        # 复制标签
        label_name = os.path.splitext(f)[0] + '.txt'
        src_lbl = os.path.join(src_labels_dir, label_name)
        shutil.copy2(src_lbl, lbl_dst)

# 执行复制
copy_files(train_files, dst_images_train, dst_labels_train)
copy_files(val_files, dst_images_val, dst_labels_val)

# 打印统计信息
print(f"总有效文件数: {len(image_files)}")
print(f"训练集数量: {len(train_files)} ({len(train_files)/len(image_files):.1%})")
print(f"验证集数量: {len(val_files)} ({len(val_files)/len(image_files):.1%})")
print("文件划分完成！")