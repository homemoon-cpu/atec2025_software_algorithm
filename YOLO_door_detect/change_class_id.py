import os
import shutil

# 原始标签路径和备份路径
label_dir = "/home/admin/atec_project/YOLO_door_detect/datasets/labels/val"
backup_dir = os.path.join(label_dir, "labels_backup")

# 创建备份目录（如果不存在）
os.makedirs(backup_dir, exist_ok=True)

# 遍历所有标签文件
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(label_dir, filename)
        
        # 备份原文件
        backup_path = os.path.join(backup_dir, filename)
        shutil.copy2(filepath, backup_path)
        
        # 读取并修改内容
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print(f"警告：{filename} 是空文件，跳过处理")
            continue
            
        # 处理第一行
        first_line = lines[0].strip().split()
        if len(first_line) < 5:
            print(f"警告：{filename} 第一行格式错误，跳过处理")
            continue
            
        # 修改类别 ID
        first_line[0] = '4'
        modified_line = ' '.join(first_line) + '\n'
        
        # 组合新内容（第一行修改 + 后续原内容）
        new_lines = [modified_line] + lines[1:]
        
        # 写入修改后的内容
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
            
print("处理完成！原始文件已备份至:", backup_dir)