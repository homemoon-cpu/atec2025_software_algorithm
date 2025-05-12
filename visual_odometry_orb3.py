import cv2
import numpy as np
import base64

class VisualOdometryORB3:
    def __init__(self):
        self.myinfo = {'picked': 0}
        
        # 视觉里程计相关初始化
        self.prev_frame = None          # 存储前一帧的灰度图像
        self.prev_kp = None             # 前一帧的关键点
        self.prev_des = None            # 前一帧的描述子
        self.pose = np.eye(4)           # 当前位姿（4x4变换矩阵）
        
        # 假设相机内参（需要根据实际相机校准参数修改）
        self.camera_matrix = np.array([
            [500, 0, 320],   # 焦距fx=500, fy=500，光心(cx,cy)=(320,240)
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 初始化ORB特征检测器
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # 初始化BFMatcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def update_pose(self, current_frame):
        """核心位姿更新逻辑"""
        # 转换为灰度图
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 特征检测与描述子计算
        kp_current, des_current = self.orb.detectAndCompute(current_gray, None)
        
        if self.prev_des is not None and des_current is not None:
            # 特征匹配
            matches = self.bf.match(self.prev_des, des_current)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) > 20:
                # 提取匹配点坐标
                prev_pts = np.array([self.prev_kp[m.queryIdx].pt for m in matches[:50]], 
                                   dtype=np.float32).reshape(-1, 1, 2)
                curr_pts = np.array([kp_current[m.trainIdx].pt for m in matches[:50]], 
                                   dtype=np.float32).reshape(-1, 1, 2)
                
                # 计算本质矩阵
                E, mask = cv2.findEssentialMat(
                    prev_pts, curr_pts,
                    self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )
                
                # 恢复相对运动
                _, R, t, _ = cv2.recoverPose(
                    E, prev_pts, curr_pts,
                    cameraMatrix=self.camera_matrix
                )
                
                # 构建变换矩阵
                transformation = np.eye(4)
                transformation[:3, :3] = R
                transformation[:3, 3] = t.ravel()  # 注意单目尺度不确定性

                # 更新全局位姿
                self.pose = self.pose @ transformation
                
                # 存储位姿信息（示例存储简化的旋转和平移）
                self.myinfo={
                    'rotation': R,
                    'translation': t,
                    'global_pose': self.pose.copy()
                }
        
        # 更新前一帧信息
        self.prev_frame = current_gray
        self.prev_kp = kp_current
        self.prev_des = des_current

        return self.pose
    
