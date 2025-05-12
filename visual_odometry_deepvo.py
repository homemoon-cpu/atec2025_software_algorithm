import torch
import cv2
import numpy as np
from DeepVO.model import DeepVO  # 示例模型，需替换实际选用框架

class VisualOdometryDeep:
    def __init__(self, model_path, img_size=(608, 184)):
        # 初始化深度学习模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = self._load_model(model_path, img_size)
        self.img_size = img_size
        
        # 状态变量
        self.last_frame = None
        self.pose = np.eye(4)  # 初始位姿矩阵
        self.relative_pose = np.eye(4)  # 当前帧相对位姿

        # 预处理参数
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _load_model(self, model_path,img_size):
        """加载预训练视觉里程计模型"""
        try:
            model = DeepVO(img_size[0],img_size[1], batchNorm=True)
            # 兼容性加载方式
            checkpoint = torch.load(
                model_path,
                map_location='cpu'  # 先加载到CPU
            )
            checkpoint = {k: v.float() for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            
            # 转移到可用设备
            model = model.to(self.device).float()
            model.eval()
            return model
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            exit(1)

    def _preprocess(self, image):
        """图像预处理"""
        # 尺寸调整与归一化
        image = cv2.resize(image, self.img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()

    def estimate_pose(self, current_frame):
        """估计相对位姿变化"""
        with torch.no_grad():
            # 双帧输入模式
            if self.last_frame is not None:
                # # 构建模型输入 [batch, 2, C, H, W]
                # input_tensor = torch.cat([
                #     self._preprocess(self.last_frame),
                #     self._preprocess(current_frame)
                # ], dim=1).to(self.device)
                 # 确保预处理输出形状正确
                frame1 = self._preprocess(self.last_frame)  # 应为 [1, C, H, W]
                frame2 = self._preprocess(current_frame)
                
                # 显式构建序列维度
                input_tensor = torch.stack([frame1, frame2], dim=1)  # [1, 2, C, H, W]

                # 模型推理
                output = self.model(input_tensor)
                
                # 解析输出为SE3位姿
                self.relative_pose = self._output_to_se3(output)
                
            # 更新参考帧
            self.last_frame = current_frame.copy()
            return self.relative_pose

    def _output_to_se3(self, output):
        """将模型输出转换为SE3位姿矩阵"""
        # 假设模型输出6DoF参数 [tx, ty, tz, rx, ry, rz]
        # 此处需要根据实际模型输出结构调整
        pose_vector = output[0, 0, :].cpu().numpy()
        translation = pose_vector[:3]
        rotation = pose_vector[3:]
        
        # 转换为旋转矩阵
        R = cv2.Rodrigues(rotation)[0]
        
        # 构建位姿矩阵
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = translation
        return pose
    
class PoseTracker:
    def __init__(self, model_path):
        """ 初始化视觉里程计和状态变量 """
        self.vo = VisualOdometry(model_path)          # 创建视觉里程计实例
        self.global_pose = np.eye(4)                 # 全局位姿（世界坐标系到相机的变换）
        self.trajectory = [self.global_pose.copy()]  # 轨迹存储列表

    def update_pose(self, frame):
        """
        处理新帧并更新位姿
        :param frame: 输入的BGR图像帧
        :return: 更新后的全局位姿(4x4 numpy矩阵)
        """
        # 估计当前帧的相对位姿变化
        relative_pose = self.vo.estimate_pose(frame)
        
        # 更新全局位姿（矩阵相乘顺序根据坐标系定义确定）
        self.global_pose = self.global_pose @ relative_pose
        
        # 记录轨迹
        self.trajectory.append(self.global_pose.copy())
        return self.global_pose

    def get_current_position(self):
        """ 从全局位姿中提取三维坐标 """
        return self.global_pose[:3, 3].flatten()

    def visualize_trajectory(self):
        """ 实时绘制三维轨迹 """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        positions = np.array([pose[:3,3] for pose in self.trajectory])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:,0], positions[:,1], positions[:,2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def save_trajectory(self, file_path):
        """ 保存轨迹到文件 """
        with open(file_path, 'w') as f:
            for idx, pose in enumerate(self.trajectory):
                np.savetxt(f, pose, fmt='%.6f', 
                          header=f"Frame {idx} Pose Matrix")