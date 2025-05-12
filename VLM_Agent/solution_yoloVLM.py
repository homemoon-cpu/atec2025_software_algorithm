import argparse
import cv2
import time
import numpy as np
import os
import json
from VLM_Agent.agent_VLM import agent
from ultralytics import YOLO
import torch
import base64
from VLM_Agent.odometry import Odometry

class AlgSolution:
    def __init__(self):
        os.system("OLLAMA_MODELS=./checkpoints /usr/local/bin/ollama serve &")
        time.sleep(5)
        if os.path.exists('/home/admin/workspace/job/logs/'):
            self.handle = open('/home/admin/workspace/job/logs/user.log', 'w')
        else:
            self.handle = open('user.log', 'w')
        self.handle.write("ollama loaded\n")

        try:
            # 尝试使用 CUDA
            if torch.cuda.is_available():
                device = 'cuda'
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                print("CUDA is working properly")
            else:
                device = 'cpu'
                print("CUDA not available, using CPU")
        except Exception as e:
            print(f"Error initializing CUDA: {e}")
            device = 'cpu'
            print("Falling back to CPU")

        # Initialize VLM agent
        self.vlm_agent = agent()
        
        # Initialize detection model
        self.yolo_model = YOLO('checkpoints/yolo11x.pt')
        self.device = device
        
        # State tracking
        self.person_detected = False
        self.stretcher_detected = False
        self.current_target = None  # 'person' or 'stretcher'
        
        self.myinfo = {'picked': 0}
        self.prev = {
            'angular': 0,
            'velocity': 0,
            'viewport': 0, 
            'interaction': 0,
        }

        self.foreward = ([0,50],0,0)

        self.backward = ([0,-50],0,0)
        self.turnleft = ([-20,0],0,0)
        self.turnright = ([20,0],0,0)
        self.carry = ([0,0],0,3)
        self.drop = ([0,0],0,4)
        self.noaction = ([0,0],0,0)

        self.useYOLO = True 

        self.odometry = Odometry()

        self.last_action_time = time.time()


    def predicts(self, ob, success):
        current_time = time.time()
        delta_time = current_time - self.last_action_time
        self.last_action_time = current_time
        if self.prev['interaction'] == 3 and success:
            self.myinfo = {'picked': 1}

        ob = base64.b64decode(ob)
        ob = cv2.imdecode(np.frombuffer(ob, np.uint8), cv2.IMREAD_COLOR)
        pred = self.predict(ob, self.myinfo)
        res = {
            'angular': pred[0][0],
            'velocity': pred[0][1],
            'viewport': pred[1],
            'interaction': pred[2],
        }
        self.prev = res
        print(pred)
    
        # self.odometry.update(pred, delta_time)
        # self.odometry.log_status()
        return res



    def reset(self, reference_text, reference_image):
        reference_image=base64.b64decode(reference_image)
        reference_image = cv2.imdecode(np.frombuffer(reference_image, np.uint8), cv2.IMREAD_COLOR)
        # print('reset obs shape after decode', reference_image.shape)
        self.vlm_agent.reset(reference_text, reference_image)
        self.person_detected = False
        self.stretcher_detected = False
        self.current_target = None
        self.last_action_time = time.time()
        self.myinfo = {'picked': 0}


    def draw_bbox_on_obs(self,obs, boxes, labels, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on the observation image.

        Args:
            obs: The observation image (numpy array).
            boxes: List of bounding boxes, each in the format [x, y, w, h].
            labels: List of labels corresponding to the bounding boxes.
            color: Color of the bounding box (default: green).
            thickness: Thickness of the bounding box lines (default: 2).
        """
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(obs, top_left, bottom_right, color, thickness)
            cv2.putText(obs, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return obs

    def predict(self, obs, info):
        # First try to detect objects using YOLO
        if self.useYOLO:
            print(info['picked'])
            if info['picked'] == 1:
                results = self.yolo_model(source=obs, imgsz=640, conf=0.1)
            else:
                results = self.yolo_model(source=obs, imgsz=640, conf=0.2)

            boxes = results[0].boxes  # get all detected bounding box

            boxes_tmp = [box.xywh[0].tolist() for box in boxes]
            labels_tmp = [self.yolo_model.names[int(box.cls.item())] for box in results[0].boxes]

            # Draw bounding boxes on the observation image
            obs_with_bbox = self.draw_bbox_on_obs(obs, boxes_tmp, labels_tmp)
            cv2.imshow('Observation with BBox', obs_with_bbox)
            cv2.waitKey(1)
            # Check for person and stretcher in detections
            person_box = None
            stretcher_box = None
            truck_box=None
            if info['picked']==1:
                self.person_detected=True
            for box in boxes:
                cls = int(box.cls.item())
                if( self.yolo_model.names[cls] == 'person' or
                    self.yolo_model.names[cls] == 'motorcycle' or 
                    self.yolo_model.names[cls] == 'dog' or 
                    self.yolo_model.names[cls] == 'teddy bear' or 
                    self.yolo_model.names[cls] == 'bicycle'
                    ) and not self.person_detected:
                    
                    bbox = box.xywh[0].tolist()
                    x, y, w, h = [int(v) for v in bbox]
                    
                    x_min = max(0, int(x - w/2))
                    y_min = max(0, int(y - h/2))
                    x_max = min(obs.shape[1], int(x + w/2))
                    y_max = min(obs.shape[0], int(y + h/2))
                    
                    # 安全性检查
                    if x_min >= x_max or y_min >= y_max:
                        continue
                        
                    roi = obs[y_min:y_max, x_min:x_max]
                    
                    # 转为灰度图像
                    if len(roi.shape) == 3:
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    else:
                        roi_gray = roi
                    
                    # 计算灰度图像的统计数据
                    mean_gray = np.mean(roi_gray)
                    std_gray = np.std(roi_gray)
                    
                    # 显示调试信息
                    print(f"Detection: {self.yolo_model.names[cls]}, Mean gray: {mean_gray:.1f}, Std: {std_gray:.1f}")
                    
                    is_shadow = False
                
                    if len(roi.shape) == 3:  # 确保是彩色图像
                        # 计算RGB通道的标准差和均值
                        b_std = np.std(roi[:,:,0])
                        g_std = np.std(roi[:,:,1])
                        r_std = np.std(roi[:,:,2])
                        
                        b_mean = np.mean(roi[:,:,0])
                        g_mean = np.mean(roi[:,:,1])
                        r_mean = np.mean(roi[:,:,2])
                        
                        print(f"Color std: R={r_std:.1f}, G={g_std:.1f}, B={b_std:.1f}")
                        print(f"Color mean: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
                        
                        # 计算颜色通道间的比例关系
                        # 影子通常在三个通道上的比例关系更一致
                        max_channel_mean = max(r_mean, g_mean, b_mean)
                        min_channel_mean = min(r_mean, g_mean, b_mean)
                        if max_channel_mean > 0:
                            channel_ratio = min_channel_mean / max_channel_mean
                        else:
                            channel_ratio = 0
                            
                        # 计算饱和度 - 影子通常饱和度低
                        saturation = np.std([r_mean, g_mean, b_mean])
                        
                        # 根据实际影子数据调整的检测规则
                        # 1. 影子通常三个通道比较接近 (通道比率高)
                        # 2. 影子通常饱和度较低
                        # 3. 考虑到观察数据中灰度均值和颜色标准差
                        
                        # 结合以下条件判断是否为影子
                        is_shadow = (
                            # 通道比率条件 - 影子各通道比率更接近
                            (channel_ratio > 0.75) and
                            
                            # 饱和度条件 - 影子饱和度低
                            (saturation < 20) and
                            
                            # 颜色变化条件 - 考虑观察到的影子数据
                            (max(r_std, g_std, b_std) < 85) and
                            
                            # 考虑所有颜色通道的均值分布
                            (r_mean < 120 and g_mean < 140 and b_mean < 120)
                        )
                        
                        # 显示额外调试信息
                        print(f"Channel ratio: {channel_ratio:.2f}, Saturation: {saturation:.2f}")
                        print(f"Is shadow assessment: {is_shadow}")
                    
                    if not is_shadow:
                        person_box = bbox
                    
                elif self.yolo_model.names[cls] == 'suitcase' and self.person_detected:
                    stretcher_box = box.xywh[0].tolist()
                elif self.yolo_model.names[cls] == 'skateboard'  and self.person_detected:
                    stretcher_box = box.xywh[0].tolist()
                elif self.yolo_model.names[cls] =='truck'and self.person_detected:
                    truck_box = box.xywh[0].tolist()
                elif self.yolo_model.names[cls] =='refrigerator'and self.person_detected:
                    truck_box = box.xywh[0].tolist()
                elif self.yolo_model.names[cls] =='bus'and self.person_detected:
                    truck_box = box.xywh[0].tolist()

                

            # If we have a detection, use detection-based movement
            if person_box and not self.person_detected:
                return self._move_based_on_detection(person_box, 'person')
            elif stretcher_box and self.person_detected:
                return self._move_based_on_detection(stretcher_box, 'stretcher')
            elif truck_box and self.person_detected:
                return self._move_based_on_detection(truck_box, 'truck')

            
            # if not person_box == None:

            # If no detection, use VLM agent for exploration
            print('call VLM for inference...')
            return self.vlm_agent.predict(obs, info)

    def _move_based_on_detection(self, box, target_type):
        x0, y0, w_, h_ = box
        
        if target_type == 'person':
            # if w_ > h_:
            print('Detected Person')
            print(x0, y0, w_, h_)
            # if y0 - 0.5*h_ > 390 and x0>220 and x0<420:
            if y0 - 0.5*h_ > 390 :
                # self.person_detected = True
                print('carry')
                return self.carry

            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else: 
                    return ([0,40],0,0)
        elif target_type =='stretcher':
            if y0 - 0.5*h_ > 350  and x0>220 and x0<420 :
                print('drop')
                return self.drop
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward
        elif target_type == 'truck':  # stretcher
            if (w_> 320 and h_>320)  and x0>220 and x0<420 :
                print('drop')
                return self.drop
            else:
                if x0 < 220:
                    return self.turnleft
                elif x0 > 420:
                    return self.turnright
                else:
                    return self.foreward


