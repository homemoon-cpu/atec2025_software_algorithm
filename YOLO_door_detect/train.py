import os
from ultralytics import YOLO

def train_yolo_model():
    # ========== 配置参数 ==========
    config = {
        "model": "/home/admin/atec_project/checkpoints/yolo_detect.pt",       # 预训练模型 (可选 yolov8s/m/l/x)
        "data_yaml": "/home/admin/atec_project/YOLO_door_detect/door.yaml",  # 数据集配置文件
        "epochs": 10,               # 训练轮次
        "imgsz": 640,                # 输入图像尺寸
        "batch": 4,                 # 批大小 (根据GPU显存调整)
        "device": "cuda:0",            # 使用GPU ("cpu" 或 "cuda:0,1")
        "workers": 4,                # 数据加载线程数
        "name": "door_detection_v1", # 训练任务名称
        "optimizer": "auto",         # 优化器 (auto/SGD/Adam/AdamW)
        "lr0": 0.01,                 # 初始学习率
        "save": True,                # 保存训练结果
        "exist_ok": False            # 覆盖同名训练任务
    }

    # ========== 模型加载 ==========
    model = YOLO(config["model"])  # 加载预训练模型

    # ========== 开始训练 ==========
    results = model.train(
        data=config["data_yaml"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        workers=config["workers"],
        project="runs/detect",
        name=config["name"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        save=config["save"],
        exist_ok=config["exist_ok"]
    )

if __name__ == "__main__":
    train_yolo_model()