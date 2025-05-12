import os
import argparse

def clean_unlabeled_images(images_dir, labels_dir, img_ext='.jpg', label_ext='.txt'):
    """
    删除没有对应标注文件的图像文件
    :param images_dir: 图像目录路径
    :param labels_dir: 标签目录路径
    :param img_ext: 图像文件扩展名
    :param label_ext: 标签文件扩展名
    """
    # 获取文件名集合（不带扩展名）
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(img_ext)}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(label_ext)}

    # 计算需要删除的文件差集
    to_delete = image_files - label_files

    # 执行删除操作
    deleted_count = 0
    for filename in to_delete:
        img_path = os.path.join(images_dir, f"{filename}{img_ext}")
        try:
            os.remove(img_path)
            print(f"已删除无标注文件: {img_path}")
            deleted_count += 1
        except Exception as e:
            print(f"删除失败 {img_path}: {str(e)}")

    print(f"\n操作完成！共删除 {deleted_count} 个无标注图像文件")

if __name__ == "__main__":
    # 配置路径参数
    BASE_DIR = "/home/admin/atec_project/YOLO_door_detect/datasets"
    IMAGE_TRAIN_DIR = os.path.join(BASE_DIR, "images/caiji")
    LABEL_TRAIN_DIR = os.path.join(BASE_DIR, "labels/caiji")

    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='清理无标注训练图像')
    parser.add_argument('--dry-run', action='store_true', help='仅显示待删除文件，不实际执行')
    args = parser.parse_args()

    # 安全验证
    if not all(os.path.isdir(d) for d in [IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR]):
        raise ValueError("目录路径配置错误，请检查路径是否存在")

    # 执行清理（演示模式或真实操作）
    if args.dry_run:
        print("=== 演示模式（不实际删除文件）===")
        image_set = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_TRAIN_DIR) if f.endswith('.jpg')}
        label_set = {os.path.splitext(f)[0] for f in os.listdir(LABEL_TRAIN_DIR) if f.endswith('.txt')}
        missing = image_set - label_set
        print(f"发现 {len(missing)} 个待删除文件：")
        print("\n".join([f"{name}.jpg" for name in missing]))
    else:
        clean_unlabeled_images(IMAGE_TRAIN_DIR, LABEL_TRAIN_DIR)