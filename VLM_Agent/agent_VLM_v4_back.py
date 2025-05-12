from VLM_Agent.api_yoloVLM import *
from VLM_Agent.prompt_yoloVLM_v4 import *
import os
import re
import argparse
import gym
import cv2
import time
import numpy as np
import base64
from PIL import Image
import io
from openai import OpenAI
from datetime import datetime
from collections import deque
import random
import math     
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity


class ExplorationMemory:
    def __init__(self, 
                 image_size=(640, 480),
                 similarity_threshold=0.7,
                 compare_top_k=5,
                 feature_dim=512):
        # 记忆存储结构：包含图像、特征向量和空间标签
        self.history = []  
        self.image_size = image_size
        self.similarity_threshold = similarity_threshold
        self.compare_top_k = compare_top_k
        self.feature_extractor = self._build_feature_extractor(feature_dim)
        
    def _build_feature_extractor(self, output_dim):
        """构建轻量级特征提取模型（示例结构）"""
        model = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt',  # 需提供模型定义文件
            'weights.caffemodel'  # 需提供预训练权重
        )
        return lambda img: self._extract_features(img, model, output_dim)

    def _extract_features(self, img, model, output_dim):
        """使用CNN提取语义特征"""
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (224, 224)), 
            1.0, (224, 224), (104.0, 177.0, 123.0)
        )
        model.setInput(blob)
        return model.forward().flatten()[:output_dim]

    def _classify_space(self, img):
        """空间语义分类（示例实现）"""
        # 实际应替换为真正的场景分类模型
        class_mapping = {
            0: "客厅",
            1: "卧室",
            2: "厨房",
            3: "走廊"
        }
        return np.random.choice(list(class_mapping.values()))  # 随机示例

    def _annotate_image(self, img, label):
        """在图像上标注空间类型"""
        annotated = img.copy()
        cv2.putText(annotated, label, 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        return annotated

    def update_memory(self, current_frame):
        """智能更新记忆的核心方法"""
        # 预处理
        processed = cv2.resize(current_frame, self.image_size)
        features = self.feature_extractor(processed)
        
        # 动态对比策略
        is_new = True
        compare_set = self.history[-self.compare_top_k:] if self.compare_top_k else self.history
        
        for mem in compare_set:
            similarity = cosine_similarity([features], [mem['features']])[0][0]
            if similarity > self.similarity_threshold:
                is_new = False
                break

        # 处理新空间
        if is_new:
            label = self._classify_space(processed)
            self.history.append({
                'image': self._annotate_image(processed, label),
                'features': features,
                'label': label
            })
        return is_new

    def get_visual_context(self, max_length=10):
        """生成带语义标注的探索历史"""
        recent = self.history[-max_length:]
        return cv2.hconcat([item['image'] for item in recent])

    def encode_image_array(self, image_array):
        """优化图像编码"""
        success, buffer = cv2.imencode('.webp', image_array, [cv2.IMWRITE_WEBP_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    def query_vlm(self, prompt):
        """增强的VLM查询接口"""
        context = self.get_visual_context()
        return call_api_vlm(
            prompt=prompt,
            base64_image=self.encode_image_array(context),
            history_labels=[m['label'] for m in self.history]
        )


class agent:

    def __init__(self):
        if os.path.exists('/home/admin/workspace/job/logs/'):
            self.handle = open('/home/admin/workspace/job/logs/user.log', 'w')
        else:
            self.handle = open('user.log', 'w')
        self.handle.write("yoloVLM agent initialize\n")
        self.handle.flush()
        self.clue = ""
        self.landmark = []
        self.landmark_back = []
        self.target = ""
        self.action = ""
        
        # Initialize obs and info
        self.obs = None
        self.info = None
        
        self.phase = 0
        self.initialized = False
        
        # Create buffers for actions and observations
        self.action_buffer = []
        self.obs_buffer = []
        self.image_buffer = []
        
        # State tracking
        self.current_step = 0
        self.move_fail_count = 0
        self.search_in_progress = False
        self.move_to_landmark_in_progress = False
        self.observe_in_progress = False
        self.move_obstacle_in_progress = False
        self.search_move_in_progress = False
        self.pending_action = None
        self.obs_storage_indices = []
        self.around_image = None
        
        # Add step counters for limiting move_to_landmark iterations
        self.move_steps = 0
        self.max_move_steps = 4  # Match original code limit of 2 iterations
        
        # refenrence image
        self.image = None

        self.initial_direction = None

        self.start_search_from_initial = False
    
        self.search_steps = 0

        self.first_enter_return_phase = False

        self.search_move_count = 0

        self.is_indoor = False

        self.move_to_the_door_progress = False

        self.need_to_verify_indoor = True

        self.exploration_in_progress = False

        self.have_explored = set()

        self.space_layout = []

        self.no_door_count = 0

        self.call_move_to_the_door_count = 0

        

    def predict(self, obs, info):

        # Add a 1-second delay at the beginning of predict
        time.sleep(1)
        # self.handle.write(str(self.phase))
        # self.handle.write(str(self.landmark))
        # self.handle.write(str(self.target))
        self.current_step +=1
        
        # Store the current observation and info
        self.obs = obs
        self.info = info
        
        # Initialize if not done already
        if not self.initialized:
            self.initialized = True
            return ([0, 0], 0, 0)  # Initial action
        
        print(self.action_buffer)
        # Check if there are pending actions in the buffer
        if self.action_buffer:
            action = self.action_buffer.pop(0)
            
            # If this action should store an observation, mark it
            if self.obs_buffer and len(self.obs_buffer) > 0:
                if self.obs_buffer[0]:
                    # Store the current observation for future use
                    self.image_buffer.append(obs.copy())
                self.obs_buffer.pop(0)
            
            return action
        
        # Start the main logic chain if no pending actions
        if self.phase == 0:
            # Initialize by analyzing the clue
            return self._handle_initial_phase()
            
        elif self.phase == 1:
            # Search for the injured person
            return self._handle_search_phase()
            
        elif self.phase == 2:
            # Rescue the injured person
            return self._handle_rescue_phase()
            
        elif self.phase == 3:
            # Search for the stretcher/ambulance
            return self._handle_return_phase()
            
        elif self.phase == 4:
            # Place the person on the stretcher
            return self._handle_placement_phase()
        
        # Default action if nothing else to do
        return ([0, 0], 0, 0)

    def _handle_initial_phase(self):
        # First time through, analyze the clue
        if not self.landmark:
            self.analyse_initial_clue()
            self.handle.write(f"LANDMARK LIST: {', '.join(self.landmark)}\n")
            self.handle.flush()
            # Directly search in the initial direction without rotating
            
            # prompt = search_prompt_begin(self.landmark)
            # max_retries = 3
            # for attempt in range(max_retries):
            #     try:
            #         res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
            #         self.handle.write(f"[SEARCH] \n {res} \n\n\n")
            #         self.handle.flush()
            #         pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
            #         match = pattern.search(res)
                
            #         if match:
            #             analysis = match.group(1).strip()
            #             landmark = match.group(2).strip()

            #             if landmark != "None" and landmark != "NONE":
            #                 self.target = landmark
            #                 self.phase = 1
            #                 self.move_to_landmark_in_progress = True
            #                 # Queue up the initial forward movement
            #                 self.action_buffer.append(([0, 100], 0, 0))
            #                 self.action_buffer.append(([0, 100], 0, 0))
            #                 self.action_buffer.append(([0, 100], 0, 0))
            #                 return ([0, 0], 0, 0)
            #             else:
            #                 self.phase = 1
            #                 self.search_move_in_progress = True
            #                 self.start_search_from_initial = True
            #                 return self._start_search_move()

            #     except Exception as e:
            #         if attempt == max_retries - 1:
            #             raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")
            
            # If we get here, no landmark was found
            self.phase = 1
            self.move_to_the_door_progress = True
            return ([0,0],0,0)
        
        # Return default action if already initialized
        return ([0, 0], 0, 0)
    
        
    def _handle_search_phase(self):
        if self.need_to_verify_indoor:
            if self._verify_door_entry():
                self.need_to_verify_indoor = False
                self.exploration_in_progress = True
            else:
                self.exploration_in_progress = False
                self.move_to_the_door_progress = True
                exec_open_door =  self.move_to_the_door()
                return ([0,0],0,0)
            
        if self.info['picked'] == 1:
            self.phase = 3

        if self.exploration_in_progress:
            print("exploration")
            has_target = self._process_exploration_result()
            if has_target:
                print('has target')
                if 'door' in self.target:
                    exec_open_door =  self.move_to_the_door()
                    if exec_open_door:
                        self.move_to_the_door_progress = False
                        self.exploration_in_progress = True
                        return ([0,0],0,0)
                    else:
                        self.exploration_in_progress = False
                        self.move_to_the_door_progress = True
                        return ([0,0],0,0)
                else:
                    self.exploration_in_progress = False
                    self.move_to_landmark_in_progress = True
                return self._start_move_to_landmark()
            else:
                print('no target')
                self.exploration_in_progress = False
                self.search_in_progress = True
                self._process_search_move_result()
                return self._start_search()
            
        elif self.move_to_the_door_progress:
            exec_open_door =  self.move_to_the_door()
            if exec_open_door:
                self.exploration_in_progress = True
                self.move_to_the_door_progress = False
                return ([0,0],0,0)
            else:
                self.exploration_in_progress = False
                self.move_to_the_door_progress = True
                return ([0,0],0,0)
        
        elif self.search_in_progress:
            self.phase = 1
            print('search')
            self._process_search_result()
            self.search_in_progress = False
            self.exploration_in_progress = True
            return ([0,0],0,0)
            
        elif self.move_to_landmark_in_progress:
            print("move_to_landmark")
            result = self._process_move_result()
            
            if result == False:
                self.move_fail_count += 1
                self.move_to_landmark_in_progress = False
                self.exploration_in_progress = True
                has_target = self._process_exploration_result()
                if has_target:
                    print('has target')
                    if 'door' in self.target:
                        exec_open_door =  self.move_to_the_door()
                        if exec_open_door:
                            self.move_to_the_door_progress = False
                            self.exploration_in_progress = True
                            return ([0,0],0,0)
                        else:
                            self.exploration_in_progress = False
                            self.move_to_the_door_progress = True
                            return ([0,0],0,0)
                    else:
                        self.exploration_in_progress = False
                        self.move_to_landmark_in_progress = True
                    return self._start_move_to_landmark()
                else:
                    print('no target')
                    self.exploration_in_progress = False
                    self.search_in_progress = True
                    self._process_search_move_result()
                    return self._start_search()
                
            self.move_steps += 1
            if self.move_steps >= self.max_move_steps:
                self.move_to_landmark_in_progress = False
                self.exploration_in_progress = True
                has_target = self._process_exploration_result()
                if has_target:
                    print('has target')
                    if 'door' in self.target:
                        exec_open_door =  self.move_to_the_door()
                        if exec_open_door:
                            self.move_to_the_door_progress = False
                            self.exploration_in_progress = True
                            return ([0,0],0,0)
                        else:
                            self.exploration_in_progress = False
                            self.move_to_the_door_progress = True
                            return ([0,0],0,0)
                    else:
                        self.exploration_in_progress = False
                        self.move_to_landmark_in_progress = True
                    return self._start_move_to_landmark()
                else:
                    print('no target')
                    self.exploration_in_progress = False
                    self.search_in_progress = True
                    self._process_search_move_result()
                    return self._start_search()
            
            if self.target in ("injured person", "Injured person"):
                self.phase = 2
                self.move_to_landmark_in_progress = False
                self.handle.write("[SPEAK] Found injured, rescuing.\n")
                self.handle.flush()
                print("[SPEAK] Found injured, rescuing.\n")
                self.action_buffer.append(([0, 50], 0, 3))
                self.obs_buffer.append(False)
                return ([0, 50], 0, 0)
            
            return self._start_move_to_landmark()

        else:
            self.search_in_progress = True
            return self._start_search()

    def _process_exploration_result(self):
        prompt = exploration_prompt(self.have_explored, self.landmark)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                self.handle.write(f"[EXPLORATION] \n {res} \n\n\n")
                self.handle.flush()
                print(f"[EXPLORATION] \n {res} \n\n\n")
                pattern = re.compile(r'<think>(.*?)</think>\s*<space_layout>(.*?)</space_layout>\s*<current_position>(.*?)</current_position>\s*<target>(.*?)</target>\s*<side>(.*?)</side>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    space_layout = match.group(2).strip()
                    current_position = match.group(3).strip()
                    target = match.group(4).strip()
                    side = match.group(5).strip()

                    self.space_layout.append(space_layout)
                    self.have_explored.add(current_position)

                    if target != "None": 
                        self.target = target
                        
                        if side == "left":
                            print("left")
                            for _ in range(1):
                                self.action_buffer.append(([-30, 0], 0, 0))
                                self.obs_buffer.append(False)
                                self.action_buffer.append(([0, 50], 0, 0))
                                self.obs_buffer.append(False)
                        elif side == "right":
                            print("right")
                            for _ in range(1):
                                self.action_buffer.append(([30, 0], 0, 0))
                                self.obs_buffer.append(False)
                                self.action_buffer.append(([0, 50], 0, 0))
                                self.obs_buffer.append(False)
                        elif side == "center":
                            print("center")
                            self.action_buffer.append(([0, 100], 0, 0))
                            self.obs_buffer.append(False)
                        return True    
                    elif target == 'None':
                        return False
            except Exception as e:
                self.handle.write(f"Error in exploration: {str(e)}\n")
                self.handle.flush()


    def _check_search_move_count(self):
        """处理search_move计数，达到15次后触发search_in_progress"""
        self.search_move_count += 1
        if self.search_move_count >= 15:
            self.search_move_count = 0  # 重置计数器
            self.search_in_progress = True
            self.search_move_in_progress = False
            return self._start_search()
        else:
            self.search_move_in_progress = True
            return self._start_search_move()
    
    def _handle_rescue_phase(self):
        # If person is picked, move to next phase
        if self.info['picked'] == 1:
            self.phase = 3
            self.handle.write("[SPEAK] I have rescued them. It's time to go back.\n")
            self.handle.flush()
            print("[SPEAK] I have rescued them. It's time to go back.\n")
            return ([0, 0], 0, 0)
            
        # If moving to landmark is in progress
        elif self.move_to_landmark_in_progress:
            result = self._process_move_result()
            
            if result == False:
                self.move_to_landmark_in_progress = False
                self.phase = 1
                self.target = ""
                self.handle.write("[SPEAK] I lost the injured person. Let me try to search again.\n")
                self.handle.flush()
                print("[SPEAK] I lost the injured person. Let me try to search again.\n")
                return ([0, 20], 0, 0)
                
            # Continue moving to landmark
            return self._start_move_to_landmark()
            
        # Start moving to landmark
        else:
            self.move_to_landmark_in_progress = True
            self.move_steps = 0  # Reset step counter
            return self._start_move_to_landmark()
    
    def _handle_return_phase(self):
        # Update landmarks for the return journey
        self.handle.write('enter return phase\n')
        self.handle.flush()
        print("enter return phase\n")
        if self.first_enter_return_phase == False:
            self.first_enter_return_phase = True
        self.landmark = ['ambulance', 'stretcher']
        
        # If search is in progress
        if self.search_in_progress:
            result = self._process_search_result()
            self.search_in_progress = False
            
            if self.target == "stretcher" or self.target == "Stretcher":
                self.phase = 4
                self.handle.write("[SPEAK] I found the stretcher. Let me place the injured person on it.\n")
                self.handle.flush()
                print("[SPEAK] I found the stretcher. Let me place the injured person on it.\n")
                return ([0, 50], 0, 0)
                
            if result == True:  # landmark exists in view
                if self.move_obstacle_in_progress:
                    obstacle_result = self._process_obstacle_result()
                    if obstacle_result == True:
                        self.move_obstacle_in_progress = False
                        return self._start_search()
                
                self.move_to_landmark_in_progress = True
                return self._start_move_to_landmark()
            else:
                self.search_in_progress = True
                return self._start_search()
        
        elif self.move_obstacle_in_progress:
            result = self._process_obstacle_result()
            self.move_obstacle_in_progress = False
            self.move_to_landmark_in_progress = True
            return self._start_move_to_landmark()
                
        # If moving to landmark is in progress
        elif self.move_to_landmark_in_progress:
            result = self._process_move_result()
            
            if result == False:
                self.move_to_landmark_in_progress = False
                self.search_in_progress = True
                return self._start_search()
            else:
                self.move_steps += 1
                if self.move_steps >= self.max_move_steps:
                    # If reached max steps, go back to search
                    self.move_to_landmark_in_progress = False
                    self.search_in_progress = True
                    return self._start_search()
            # Keep trying to move to landmark
            return self._start_move_to_landmark()
        
        # Start the search process
        else:
            self.search_in_progress = True
            return self._start_search()
    
    def _handle_placement_phase(self):
        # If moving to landmark is in progress
        if self.move_to_landmark_in_progress:
            result = self._process_move_result()
            
            if result == False:
                self.move_to_landmark_in_progress = False
                self.phase = 1
                # Queue backward movements and drop action
                self.action_buffer.append(([0, -80], 0, 0))
                self.action_buffer.append(([0, 0], 0, 4))
                return ([0, -80], 0, 0)
                
            # Continue moving to landmark
            return self._start_move_to_landmark()
            
        # Start moving to landmark
        else:
            self.move_to_landmark_in_progress = True
            return self._start_move_to_landmark()
        
    def move_to_the_door(self):
        print("move_to_the_door")
        if self.is_indoor:
            self.call_move_to_the_door_count+=1
            print(f"call_move_to_the_door_count: {self.call_move_to_the_door_count}")
            if self.call_move_to_the_door_count >= 10:
                self.call_move_to_the_door_count = 0
                return True
        prompt = door_prompt()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                self.handle.write(f"[Door Direction] \n {res} \n\n\n")
                self.handle.flush()
                print(f"[Door Direction] \n {res} \n\n\n")
                pattern = re.compile(r'<a>(.*?)</a>\s*<position>(.*?)</position>\s*<operable>(.*?)</operable>', re.DOTALL)
                match = pattern.search(res)
                if match:
                    analysis_text = match.group(1).strip()
                    door_direction = match.group(2).strip()
                    is_operable = match.group(3).strip()

                    if is_operable == "yes":
                        self.handle.write("[DOOR] Door is operable, attempting to open.\n")
                        self.handle.flush()
                        
                        self.action_buffer.append(([0, 0], 0, 5))  
                        self.obs_buffer.append(False)
                        
                        if self.need_to_verify_indoor:
                            self.action_buffer.append(([0, 50], 0, 0))  
                            self.obs_buffer.append(False)

                            self.action_buffer.append(([0, 100], 0, 0))  
                            self.obs_buffer.append(False)
                        
                        self.action_buffer.append(([0, 0], 0, 0))  
                        self.obs_buffer.append(True)                     
                        
                        self.need_to_verify_indoor = True
                        return True
                    
                    # 太靠近门的情况
                    if door_direction == "TOO_CLOSE":
                        self.handle.write("[DOOR] Too close to door, attempting to open.\n")
                        self.handle.flush()
                        
                        self.action_buffer.append(([0, 0], 0, 5))  # 开门交互
                        self.obs_buffer.append(False)
                        
                        if self.need_to_verify_indoor:
                            self.action_buffer.append(([0, 50], 0, 0))  # 向前移动通过门
                            self.obs_buffer.append(False)

                            self.action_buffer.append(([0, 100], 0, 0))  # 向前移动通过门
                            self.obs_buffer.append(False)
                        
                        self.action_buffer.append(([0, 0], 0, 0))  
                        self.obs_buffer.append(True)  
                        
                        self.need_to_verify_indoor = True
                        return True
                    
                    # 没有门可见，继续搜索
                    if door_direction == "NONE":
                        self.no_door_count +=1
                        if self.no_door_count % 2 == 0:
                            self.no_door_count = 0
                            self.action_buffer.append(([0, 0], 0, 5))
                            self.obs_buffer.append(False)
                            if self.need_to_verify_indoor:
                                self.action_buffer.append(([0, 50], 0, 0))  
                                self.obs_buffer.append(False)
                                self.action_buffer.append(([0, 100], 0, 0))  
                                self.obs_buffer.append(False)
                            self.action_buffer.append(([0, 0], 0, 0))  
                            self.obs_buffer.append(True)                     
                            self.need_to_verify_indoor = True
                            return True
                        # self.handle.write("[DOOR] No door detected, moving forward to search.\n")
                        # self.handle.flush()
                        # # 使用较小的步进以避免错过门
                        # self.action_buffer.append(([0, 80], 0, 0))
                        # self.obs_buffer.append(False)
                        return False
                    
                    # 门可见，根据位置调整
                    elif door_direction == "left":
                        self.handle.write("[DOOR] Door detected on left side, repositioning.\n")
                        self.handle.flush()
                        self.action_buffer.append(([-30, 0], 0, 0))  # 略微向左转
                        self.obs_buffer.append(False)
                        self.action_buffer.append(([0, 50], 0, 0))
                        self.obs_buffer.append(False)
                        return False
                    elif door_direction == "middle":
                        self.handle.write("[DOOR] Door detected in the middle, moving forward.\n")
                        self.handle.flush()
                        self.action_buffer.append(([0, 50], 0, 0))
                        self.obs_buffer.append(False)
                        return False
                    elif door_direction == "right":
                        self.handle.write("[DOOR] Door detected on right side, repositioning.\n")
                        self.handle.flush()
                        self.action_buffer.append(([30, 0], 0, 0))  # 略微向右转
                        self.obs_buffer.append(False)
                        self.action_buffer.append(([0, 50], 0, 0))
                        self.obs_buffer.append(False)
                        return False
                    
                    
                        
            except Exception as e:
                self.handle.write(f"Error in door direction detection: {str(e)}\n")
                self.handle.flush()
        
    
    def _verify_door_entry(self):
        """验证在门交互后是否已经进入建筑内部"""
        if not self.image_buffer:
            return False
        
        # 使用最后保存的观察结果检查室内状态
        verification_image = self.image_buffer[-1]
        self.image_buffer = []  # 使用后清空缓冲区
        
        try:
            # 创建一个提示词来检查我们是否在室内
            prompt = """
            Carefully analyze this image and determine if it shows an indoor environment.

            Look for the following features:
            1. Walls, ceilings, indoor lighting fixtures
            2. Indoor furniture or decorations 
            3. Confined space typical of building interiors
            4. Absence of outdoor elements (open sky, large trees/vegetation)
            5. Interior doorways, hallways, or room structure

            Please respond with ONLY ONE of these exact terms:
            - "INDOOR" - if you're confident this shows an interior environment
            - "OUTDOOR" - if you're confident this shows an exterior environment  
            - "UNCERTAIN" - if you cannot determine with confidence

            No explanation or additional text.
            """
            
            res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(verification_image))
            self.handle.write(f"[DOOR VERIFICATION] \n {res} \n\n\n")
            self.handle.flush()
            
            if "INDOOR" in res.upper():
                self.handle.write("[DOOR] Successfully entered building. Beginning indoor search.\n")
                self.handle.flush()
                self.move_to_the_door_progress = False
                self.search_in_progress = True
                self.is_indoor = True
                self.need_to_verify_indoor = False
                return True
            elif "UNCERTAIN" in res.upper():
                # 如果不确定，再尝试向前移动并再次检查
                self.handle.write("[DOOR] Uncertain if inside - moving forward more.\n")
                self.handle.flush()
                self.action_buffer.append(([0, 10], 0, 0))  # 再向前移动
                self.action_buffer.append(([0, 0], 0, 5))
                self.obs_buffer.append(True)  # 保存下一个观察结果以便再次检查
                self.need_to_verify_indoor = True  # 保持验证标志
                return False
                
        except Exception as e:
            self.handle.write(f"Error verifying door entry: {str(e)}\n")
            self.handle.flush()
            return False

    # Helper methods to start various processes
    def _start_search(self):
        # Prepare for search by observing surroundings
        self.observe_in_progress = True
        self.around_image = None
        self.image_buffer = []
        
        # For phase 3 or 4, start rotation to observe surroundings
        # if self.start_search_from_initial == False:
        #     self.action_buffer = []
        #     self.obs_buffer = []
        # else:
        #     self.start_search_from_initial = False
        
        if self.phase == 3 or self.phase == 4:
            # Prepare for 4 observations while rotating
            for i in range(4):
                # For first observation, don't rotate
                if i == 0:
                    self.action_buffer.append(([0, 0], 0, 0))
                    self.obs_buffer.append(True)
                else:
                    # Add 3 rotations (first 2 don't save observations)
                    for j in range(3):
                        self.action_buffer.append(([30, 0], 0, 0))
                        # Only save observation on the last rotation of each set
                        self.obs_buffer.append(j == 2)
            for i in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)


            # Return first action
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())
            return action
        else:
            # First rotate left 3 times (don't save observations)
            for _ in range(3):
                self.action_buffer.append(([-30, 0], 0, 0))
                self.obs_buffer.append(False)
            
            # Now we're facing left - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)
            
            # Rotate right 3 times to face forward (don't save)
            for _ in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)
            
            # Now we're facing forward - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)
            
            # Rotate right 3 more times to face right (don't save)
            for _ in range(3):
                self.action_buffer.append(([30, 0], 0, 0))
                self.obs_buffer.append(False)
            
            # Now we're facing right - save this observation
            self.action_buffer.append(([0, 0], 0, 0))
            self.obs_buffer.append(True)
            
            # Rotate left 3 times to return to center (don't save)
            for _ in range(3):
                self.action_buffer.append(([-30, 0], 0, 0))
                self.obs_buffer.append(False)
            
            # Get first action and observation status
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())
            
            return action
    
    def _start_move_to_landmark(self):
        # This action will be handled immediately, the result processed on next prediction
        return ([0, 0], 0, 0)
    
    def _start_move_obstacle(self):
        # Prepare for obstacle check
        self.action_buffer = [
            ([0, -50], 0, 0),
            ([0, 0], 2, 0)
        ]
        self.obs_buffer = [False, False]
        
        # Return first action
        action = self.action_buffer.pop(0)
        self.obs_buffer.pop(0)
        return action
    
    def _start_search_move(self):
        # Prepare for search move
        self.observe_in_progress = True
        self.image_buffer = []
        
        # Set up observation sequence
        if self.start_search_from_initial == False:
            self.action_buffer = []
            self.obs_buffer = []
        else:
            self.start_search_from_initial = False
        
        if self.phase == 3 or self.phase == 4:
            # Similar to search but with a focus on movement
            for i in range(4):
                if i == 0:
                    self.action_buffer.append(([0, 0], 0, 0))
                    self.obs_buffer.append(True)
                else:
                    for j in range(3):
                        self.action_buffer.append(([30, 0], 0, 0))
                        self.obs_buffer.append(j == 2)
            
            # Get first action and observation status
            action = self.action_buffer.pop(0)
            should_save = self.obs_buffer.pop(0)
            if should_save:
                self.image_buffer.append(self.obs.copy())
            
            return action
        else:
            # First rotate left 3 times (don't save observations)
            # for _ in range(3):
            #     self.action_buffer.append(([-30, 0], 0, 0))
            #     self.obs_buffer.append(False)
            
            # # Now we're facing left - save this observation
            # self.action_buffer.append(([0, 0], 0, 0))
            # self.obs_buffer.append(True)
            
            # # Rotate right 3 times to face forward (don't save)
            # for _ in range(3):
            #     self.action_buffer.append(([30, 0], 0, 0))
            #     self.obs_buffer.append(False)
            
            # # Now we're facing forward - save this observation
            # self.action_buffer.append(([0, 0], 0, 0))
            # self.obs_buffer.append(True)
            
            # # Rotate right 3 more times to face right (don't save)
            # for _ in range(3):
            #     self.action_buffer.append(([30, 0], 0, 0))
            #     self.obs_buffer.append(False)
            
            # # Now we're facing right - save this observation
            # self.action_buffer.append(([0, 0], 0, 0))
            # self.obs_buffer.append(True)
            
            # # Rotate left 3 times to return to center (don't save)
            # for _ in range(3):
            #     self.action_buffer.append(([-30, 0], 0, 0))
            #     self.obs_buffer.append(False)
            
            # Get first action and observation status
            # action = self.action_buffer.pop(0)
            # should_save = self.obs_buffer.pop(0)
            # if should_save:
            #     self.image_buffer.append(self.obs.copy())
            
            # return action
            return ([0,0],0,0)
    
    
    def _check_environment_match(self):
        """检查当前环境与参考图像环境的匹配程度"""
        if not hasattr(self, 'image') or not self.obs:
            return 0
        
        try:
            # 构建环境比较提示词
            prompt = f"""
            Reference environment description: "{self.image}"
            
            Compare this description with the current image. On a scale of 0-100, how similar is the current environment to the described reference environment?
            
            Focus specifically on:
            - Similar terrain (grass, pavement, etc.)
            - Similar structures (buildings, walls, etc.)
            - Similar overall lighting conditions
            
            Respond with just a number from 0-100 representing the match percentage.
            """
            
            res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
            # 尝试提取数字
            match = re.search(r'(\d+)', res)
            if match:
                score = int(match.group(1))
                return score / 100.0
            else:
                return 0
                
        except Exception as e:
            self.handle.write(f"Error checking environment match: {str(e)}\n")
            self.handle.flush()
            return 0

    
    # Methods to process results of various actions
    def _process_search_result(self):
        # If we've collected observations, process them
        if self.image_buffer:
            self.around_image = self.concatenate_images(self.image_buffer)
            self.image_buffer = []
            
            if self.phase == 0:
                self.phase = 1
                prompt = search_prompt_begin(self.landmark)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                        self.handle.write(f"[SEARCH] \n {res} \n\n\n")
                        self.handle.flush()
                        pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                        match = pattern.search(res)

                        if match:
                            analysis = match.group(1).strip()
                            landmark = match.group(2).strip()

                            if landmark != "None" and landmark != "NONE":
                                self.target = landmark
                                # Queue search confirmation action
                                self.action_buffer.append(([0, 100], 0, 1))
                                self.obs_buffer.append(False)  # Don't save observation after this action
                                return True
                            else:
                                return False

                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

            elif self.phase == 3 or self.phase == 4:
                prompt = search_prompt_back(self.landmark)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
                        self.handle.write(f"[SEARCH] \n {res} \n\n\n")
                        self.handle.flush()
                        pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                        match = pattern.search(res)

                        if match:
                            analysis = match.group(1).strip()
                            landmark = match.group(2).strip()
                            side = match.group(3).strip()

                            if landmark != "None" and landmark != "NONE":
                                self.target = landmark
                                # Queue appropriate rotation actions based on side
                                if side == "left":
                                    for _ in range(3):
                                        self.action_buffer.append(([-30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                elif side == "right":
                                    for _ in range(3):
                                        self.action_buffer.append(([30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                elif side == "back":
                                    for _ in range(6):
                                        self.action_buffer.append(([30, 0], 0, 0))
                                        self.obs_buffer.append(False)  # Don't save observations during rotation
                                # Add forward action
                                self.action_buffer.append(([0, 100], 0, 0))
                                self.obs_buffer.append(False)  # Don't save observation after this action
                                return True
                            else:

                                return False

                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")

            else:
                prompt = search_prompt(self.landmark)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
                        self.handle.write(f"[SEARCH] \n {res} \n\n\n")
                        self.handle.flush()
                        print(f"[SEARCH] \n {res} \n\n\n")
                        pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                        match = pattern.search(res)

                        if match:
                            analysis = match.group(1).strip()
                            side = match.group(2).strip()

                            if side == "left":
                                for _ in range(3):
                                    self.action_buffer.append(([-30, 0], 0, 0))
                                    self.obs_buffer.append(False)  # Don't save observations during rotation
                                self.action_buffer.append(([0, 100], 0, 0))
                                self.obs_buffer.append(False)  # Don't save observation after this action    
                            elif side == "right":
                                for _ in range(3):
                                    self.action_buffer.append(([30, 0], 0, 0))
                                    self.obs_buffer.append(False)  # Don't save observations during rotation
                                self.action_buffer.append(([0, 100], 0, 0))
                                self.obs_buffer.append(False)  # Don't save observation after this action    
                            else:
                                # Add forward action
                                self.action_buffer.append(([0, 100], 0, 0))
                                self.obs_buffer.append(False)  # Don't save observation after this action

                            return True

                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")
        
        return False
    
    def _process_move_result(self):
        prompt = move_forward_prompt(self.target)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.add_vertical_lines(self.obs)))
                self.handle.write(f"[MOVING] \n {res} \n\n\n")
                self.handle.flush()
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    tag = match.group(1).strip()
                    direction = match.group(2).strip()

                    if tag == "no" or tag == "No" or tag == "NO":
                        return False
                    elif tag == "yes" or tag == "Yes" or tag == "YES":
                        # Queue appropriate actions based on direction
                        if direction == "left":
                            self.action_buffer.append(([-30, 0], 0, 0))
                            self.obs_buffer.append(False)  # Don't save observation during turning
                        elif direction == "right":
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)  # Don't save observation during turning
                        elif direction == "middle":
                            self.action_buffer.append(([0, 100], 0, 1))
                            self.obs_buffer.append(False)  # Don't save observation during forward movement
                        
                        # forward action
                        self.action_buffer.append(([0, 100], 0, 1))
                        self.obs_buffer.append(False)  # Don't save observation during forward movement
                        
                        # Special handling for injured person target
                        if (self.target == 'injured person' or self.target == 'Injured person') and self.info['picked'] == 0:
                            # Add pickup action (3) as in original code
                            self.action_buffer.append(([0, 0], 0, 3))
                            self.obs_buffer.append(False)  # Don't save observation during pickup
                        # jump can fix many bugs!
                        elif self.info['picked'] != 1:
                            self.action_buffer.append(([0, 40], 0, 1))
                            self.obs_buffer.append(False)  # Don't save observation during pickup check
                        
                        return True

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[MOVE] Failed after {max_retries} attempts: {str(e)}")
        
        return False
    
    def _process_obstacle_result(self):
        prompt = move_obstacle_prompt()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                self.handle.write(f"[CHECKING OBSTACLE] \n {res} \n\n\n")
                self.handle.flush()
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    tag = match.group(1).strip()
                    obstacle = match.group(2).strip()

                    if tag == "1":
                        # Queue random turning and movement to avoid obstacle
                        turn = random.choice([-30, 30])
                        self.action_buffer.append(([turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during turning
                        
                        self.action_buffer.append(([turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during turning
                        
                        # Add forward movements
                        for _ in range(5):
                            self.action_buffer.append(([0, 100], 0, 0))
                            self.obs_buffer.append(False)  # Don't save observations during forward movement
                        
                        # Add return rotation
                        self.action_buffer.append(([-1 * turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during return rotation
                        
                        self.action_buffer.append(([-1 * turn, 0], 0, 0))
                        self.obs_buffer.append(False)  # Don't save observation during return rotation
                        
                        return True
                    else:
                        return False

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[MOVE_OBSTACLE] Failed after {max_retries} attempts: {str(e)}")
        
        return False
    
    
    
    def _process_search_move_result(self):
        """处理搜索移动的结果，使用当前观测而不是拼接图像"""
        print('search_move')
        try:
            self.around_image = self.obs.copy()
            prompt = search_move_prompt(self.landmark)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
                    self.handle.write(f"[SEARCH-MOVE] \n {res} \n\n\n")
                    self.handle.flush()
                    print(f"[SEARCH-MOVE] \n {res} \n\n\n")
                    
                    pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                    match = pattern.search(res)

                    if match:
                        landmark = match.group(2).strip()
                        side = match.group(3).strip()

                        if side not in ["None", "NONE"]:
                            if side == "left":
                                print("left")
                                self.action_buffer.append(([-30, 0], 0, 0))
                                self.obs_buffer.append(False)
                            elif side == "right":
                                print("right")
                                self.action_buffer.append(([30, 0], 0, 0))
                                self.obs_buffer.append(False)
                            print("middle")
                            self.action_buffer.append(([0, 100], 0, 0))
                            self.obs_buffer.append(False)
                            self.target = landmark
                            return True
                        else:
                            return False
                
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.handle.write(f"[SEARCH_MOVE] Failed after {max_retries} attempts: {str(e)}\n")
                        self.handle.flush()
                        self.action_buffer.append(([0, 0], 0, 0))
                        self.obs_buffer.append(True)
                        return False
        
        except Exception as e:
            self.handle.write(f"Error in _process_search_move_result: {str(e)}\n")
            self.handle.flush()
            return False
        
        return False

    def add_visual_guides(self, image):
        """向图像添加视觉引导标记，帮助分析方向"""
        h, w, c = image.shape
        marked_image = image.copy()
        
        # 添加中心垂直线
        center_x = w // 2
        cv2.line(marked_image, (center_x, 0), (center_x, h), (0, 255, 255), 2)
        
        # 添加左右三分线
        left_third = w // 3
        right_third = w * 2 // 3
        cv2.line(marked_image, (left_third, 0), (left_third, h), (0, 0, 255), 1)
        cv2.line(marked_image, (right_third, 0), (right_third, h), (0, 0, 255), 1)
        
        # 添加方向指示文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(marked_image, "LEFT", (left_third // 2 - 30, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(marked_image, "CENTER", (center_x - 40, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(marked_image, "RIGHT", (right_third + (w - right_third) // 2 - 30, 30), font, 0.7, (0, 255, 255), 2)
        
        return marked_image
    
    # def _process_search_move_result(self):
    #     if self.image_buffer:
    #         # 拼接图像以供分析
    #         self.around_image = self.concatenate_images(self.image_buffer)
    #         h, w, c = self.around_image.shape
     
    #         labeled_panorama = self.add_panorama_labels(self.around_image)
    #         # 将参考图像添加到全景图像的顶部
    #         reference_height = 640  # 参考图像的显示高度
    #         reference_resized = cv2.resize(self.image, (w, reference_height))
    #         combined_image = np.vstack([reference_resized, np.ones((10, w, 3), dtype=np.uint8)*255, labeled_panorama])
    #         # 在参考图像上方添加"REFERENCE IMAGE"标签
    #         reference_label_image = np.ones((30, w, 3), dtype=np.uint8) * 255
    #         cv2.putText(reference_label_image, "REFERENCE IMAGE", (w//2 - 100, 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #         combined_image = np.vstack([reference_label_image, combined_image])

    #         self.image_buffer = []
    #     else:
    #         # 如果没有图像缓冲，直接使用当前观察
    #         self.around_image = self.obs.copy()
    #     prompt = search_move_prompt(self.landmark,self.person_text)
    #     max_retries = 3
    #     for attempt in range(max_retries):
    #         try:
    #             res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.around_image))
    #             self.handle.write(f"[SEARCH-MOVE] \n {res} \n\n\n")
    #             pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
    #             match = pattern.search(res)

    #             if match:
    #                 landmark = match.group(2).strip()
    #                 side = match.group(3).strip()

    #                 if side != "None" and side != "NONE" :
    #                     # Queue appropriate rotation actions based on side
    #                     if side == "left":
    #                         for _ in range(3):
    #                             self.action_buffer.append(([-30, 0], 0, 0))
    #                             self.obs_buffer.append(False)  # Don't save observations during rotation
    #                     elif side == "right":
    #                         for _ in range(3):
    #                             self.action_buffer.append(([30, 0], 0, 0))
    #                             self.obs_buffer.append(False)  # Don't save observations during rotation

    #                     self.target = landmark
    #                     return True


    #                 # if landmark != "None" and landmark != "NONE":
    #                 #     self.target = landmark  # 将检测到的路标赋值给self.target

    #                 #     return True
    #                 else:
    #                     try:
    #                         panorama_prompt = compare_reference_image()
    #                         res = call_api_vlm(prompt=panorama_prompt, base64_image=self.encode_image_array(combined_image))
    #                         self.handle.write(f"[PANORAMA COMPARISON] \n {res} \n\n\n")
                            
    #                         # 解析最佳方向
    #                         direction_pattern = re.compile(r'<best_direction>(.*?)</best_direction>', re.DOTALL)
    #                         direction_match = direction_pattern.search(res)
    #                         best_direction = "front"  # 默认前方
                            
    #                         if direction_match:
    #                             best_direction = direction_match.group(1).strip().lower()
                                
    #                         # 解析信心得分
    #                         confidence_pattern = re.compile(r'<confidence>(.*?)</confidence>', re.DOTALL)
    #                         confidence_match = confidence_pattern.search(res)
    #                         confidence = 5  # 默认中等信心
                            
    #                         if confidence_match:
    #                             try:
    #                                 confidence = float(confidence_match.group(1).strip())
    #                             except ValueError:
    #                                 confidence = 5
                            
    #                         # 如果信心足够高（>6），使用VLM推荐的方向
    #                         if confidence > 6:
    #                             self.handle.write(f"[DIRECTION DECISION] Moving {best_direction} based on panoramic comparison (confidence: {confidence}/10)")
                                
    #                             # 根据建议方向设置动作
    #                             if best_direction == "left":
    #                                 for _ in range(3):
    #                                     self.action_buffer.append(([-30, 0], 0, 0))
    #                                     self.obs_buffer.append(False)
    #                             elif best_direction == "right":
    #                                 for _ in range(3):
    #                                     self.action_buffer.append(([30, 0], 0, 0))
    #                                     self.obs_buffer.append(False)
                                
    #                             # 添加前进动作，根据信心程度调整距离
    #                             move_distance = int(50 + (confidence/10) * 50)  # 信心越高，移动距离越大
    #                             self.action_buffer.append(([0, move_distance], 0, 0))
    #                             self.obs_buffer.append(False)


    #                     except Exception as e:
    #                         self.handle.write(f"Error in panorama analysis: {str(e)}")
    #                         # 出错时继续常规搜索流程
        
                        
                    
                        
                        

    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 raise ValueError(f"[SEARCH_MOVE] Failed after {max_retries} attempts: {str(e)}")
        
    #     return False
    
    def add_panorama_labels(self, panorama_image):
        """向全景图像添加左、中、右标签和分割线"""
        h, w, c = panorama_image.shape
        segment_width = w // 3
        
        # 创建一个副本以避免修改原始图像
        labeled_image = panorama_image.copy()
        
        # 添加垂直分割线
        line1_x = segment_width
        line2_x = segment_width * 2
        cv2.line(labeled_image, (line1_x, 0), (line1_x, h), (0, 0, 255), 2)
        cv2.line(labeled_image, (line2_x, 0), (line2_x, h), (0, 0, 255), 2)
        
        # 添加文本标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 255, 255)
        thickness = 2
        
        # 左侧标签
        cv2.putText(labeled_image, "LEFT VIEW", (segment_width//2 - 50, 30),
                    font, font_scale, font_color, thickness)
        
        # 中间标签
        cv2.putText(labeled_image, "FRONT VIEW", (segment_width + segment_width//2 - 60, 30),
                    font, font_scale, font_color, thickness)
        
        # 右侧标签
        cv2.putText(labeled_image, "RIGHT VIEW", (2*segment_width + segment_width//2 - 50, 30),
                    font, font_scale, font_color, thickness)
        
        return labeled_image

    # Original helper methods preserved
    def analyse_initial_clue(self):
        prompt = initial_clue_prompt(self.clue)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_llm(prompt=prompt)
                self.handle.write(res)
                self.handle.flush()
                # 修改正则表达式以匹配新增的<d>标签
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    direction = match.group(2).strip()
                    self.initial_direction = direction
                    # direction = self.extract_direction(self.clue,['man', 'person', 'injured'])
                    # Queue appropriate rotation actions based on direction
                    if direction == "left":
                        for _ in range(3):
                            self.action_buffer.append(([-30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif direction == "right":
                        for _ in range(3):
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif direction == "back":
                        for _ in range(6):
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif "front" in direction and "left" in direction:
                        self.action_buffer.append(([-30, 0], 0, 0))
                        self.obs_buffer.append(False)
                        self.action_buffer.append(([-15, 0], 0, 0))
                        self.obs_buffer.append(False)
                    elif "front" in direction and "right" in direction:
                        self.action_buffer.append(([30, 0], 0, 0))
                        self.obs_buffer.append(False)
                        self.action_buffer.append(([15, 0], 0, 0))
                        self.obs_buffer.append(False)
                    elif ("rear" in direction or "back" in direction) and "left" in direction:
                        for _ in range (4):
                            self.action_buffer.append(([-30, 0], 0, 0))
                            self.obs_buffer.append(False)
                    elif ("rear" in direction or "back" in direction) and "right" in direction:
                        for _ in range (4):
                            self.action_buffer.append(([30, 0], 0, 0))
                            self.obs_buffer.append(False)

                    # Parse landmarks
                    self.landmark = match.group(3).strip().split(",")
                    self.landmark = ["injured person" if landmark.strip().lower() == "injured woman" or landmark.strip().lower() == "injured man"
                                else landmark.strip() for landmark in self.landmark]
                    self.landmark = [landmark for landmark in self.landmark if "stretcher" not in landmark and "ambulance" not in landmark]
                    print(self.landmark)
                    self.landmark_back = ['ambulance','stretcher'] 
                     
                    return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[CLUE ANALYSE] Failed after {max_retries} attempts: {str(e)}")
        
        
    def create_comparison_image(self, reference_base64, current_base64):
        """创建参考图像和当前图像的并排比较图"""
        try:
            # 解码参考图像
            reference_img_data = base64.b64decode(reference_base64)
            reference_img = Image.open(io.BytesIO(reference_img_data))
            
            # 解码当前图像
            current_img_data = base64.b64decode(current_base64)
            current_img = Image.open(io.BytesIO(current_img_data))
            
            # 调整大小确保两个图像高度相同
            height = min(reference_img.height, current_img.height)
            ref_width = int(reference_img.width * (height / reference_img.height))
            cur_width = int(current_img.width * (height / current_img.height))
            
            reference_img = reference_img.resize((ref_width, height))
            current_img = current_img.resize((cur_width, height))
            
            # 创建新图像
            total_width = ref_width + cur_width
            comparison_img = Image.new('RGB', (total_width, height))
            
            # 粘贴图像
            comparison_img.paste(reference_img, (0, 0))
            comparison_img.paste(current_img, (ref_width, 0))
            
            # 添加标签
            draw = ImageDraw.Draw(comparison_img)
            try:
                # 尝试加载字体
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = None
                
            draw.text((10, 10), "Reference", fill=(255, 255, 0), font=font)
            draw.text((ref_width + 10, 10), "Current", fill=(255, 255, 0), font=font)
            
            # 转换为base64
            buffered = io.BytesIO()
            comparison_img.save(buffered, format="JPEG")
            comparison_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return comparison_base64
            
        except Exception as e:
            self.handle.write(f"Error creating comparison image: {str(e)}\n")
            self.handle.flush()
            return current_base64  # 失败时返回当前图像

    def encode_image_array(self, image_array):
        # Convert the image array to a PIL Image object
        image = Image.fromarray(np.uint8(image_array))

        # Save the PIL Image object to a bytes buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the bytes buffer to Base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str

    def concatenate_images(self, image_list):
        height, width, channels = image_list[0].shape

        total_width = width * len(image_list)
        concatenated_image = np.zeros((height, total_width, channels), dtype=np.uint8)

        for i, img in enumerate(image_list):
            concatenated_image[:, i * width:(i + 1) * width, :] = img

        return concatenated_image

    def add_vertical_lines(self, image_array):
        h, w, c = image_array.shape

        line1 = w // 3
        line2 = w * 2 // 3
        line_color = (0, 0, 255)
        line_thickness = 2

        # Create a copy to avoid modifying the original
        image_copy = image_array.copy()
        cv2.line(image_copy, (line1, 0), (line1, h), line_color, line_thickness)
        cv2.line(image_copy, (line2, 0), (line2, h), line_color, line_thickness)

        return image_copy
    
    def is_empty_image(self, img):
        """安全地检查图像是否为空"""
        if img is None:
            return True
        if isinstance(img, np.ndarray):
            return img.size == 0
        if isinstance(img, bytes) or isinstance(img, str):
            return len(img) == 0
        # 对于其他类型，尝试常规布尔转换
        try:
            return not bool(img)
        except ValueError:
            # 如果遇到 ValueError（多元素数组的布尔值不明确），假设不为空
            return False
        
    def compare_direction_with_reference(self, direction_image, direction_name):
        """比较特定方向的图像与参考图像的相似度"""
        if self.is_empty_image(self.image):
            return 0.0, {}
        
        try:
            # 构建比较提示词
            prompt = f"""
                You are a rescue robot comparing your {direction_name} view with a reference image of where an injured person was located.
                
                The comparison image shows:
                - LEFT SIDE: Reference image showing where the injured person was seen
                - RIGHT SIDE: Current view in the {direction_name} direction from your position
                
                Please analyze the similarity between the current {direction_name} view and the reference image:
                1. Identify any matching landmarks or objects
                2. Estimate how similar the environments are (0-100%)
                3. Assess the probability (0-100%) that moving in this {direction_name} direction would bring you closer to the scene shown in the reference image
                
                Use XML tags to structure your response:
                <similarity_score>Percentage from 0-100</similarity_score>
                <matching_elements>List of objects/elements that appear in both images</matching_elements>
                <approach_probability>Percentage from 0-100 (probability that moving in this direction would bring you closer to the reference scene)</approach_probability>
                """
            
            # 创建对比图像，左边是参考图，右边是当前方向图
            comparison_image = self.create_comparison_image(
                self.encode_image_array(self.image), 
                self.encode_image_array(direction_image)
            )
            
            res = call_api_vlm(prompt=prompt, base64_image=comparison_image)
            self.handle.write(f"[{direction_name.upper()} COMPARISON] \n {res} \n\n\n")
            self.handle.flush()
            
            # 解析相似度评分
            score_pattern = re.compile(r'<similarity_score>(.*?)</similarity_score>', re.DOTALL)
            score_match = score_pattern.search(res)
            score = 0
            if score_match:
                score_text = score_match.group(1).strip().replace('%', '')
                try:
                    score = float(score_text) / 100.0  # 转换为0-1
                except ValueError:
                    score = 0
                    
            # 解析匹配元素
            elements_pattern = re.compile(r'<matching_elements>(.*?)</matching_elements>', re.DOTALL)
            elements_match = elements_pattern.search(res)
            matching_elements = []
            if elements_match:
                elements_text = elements_match.group(1).strip()
                matching_elements = [elem.strip() for elem in elements_text.split(',')]
            
            # 解析是否可能是目标方向
            approach_pattern = re.compile(r'<approach_probability>(.*?)</approach_probability>', re.DOTALL)
            approach_match = approach_pattern.search(res)
            approach_probability = 0
            if approach_match:
                approach_text = approach_match.group(1).strip().replace('%', '')
                try:
                    approach_probability = float(approach_text) / 100.0  # 转换为0-1
                except ValueError:
                    approach_probability = 0
            
            # 综合评分：同时考虑相似度和接近概率
            combined_score = (score * 0.4) + (approach_probability * 0.6)
            
            return score, {
                "matching_elements": matching_elements,
                "similarity_score": score,
                "approach_probability": approach_probability,
                "direction": direction_name
            }
            
        except Exception as e:
            self.handle.write(f"Error comparing {direction_name} image with reference: {str(e)}\n")
            self.handle.flush()
            return 0.0, {"direction": direction_name}
        
    def reset(self, text, image):
        self.clue = text
        self.image = image
        self.landmark = None
        self.phase = 0
        self.initialized = False
        self.action_buffer = []
        self.obs_buffer = []
        self.image_buffer = []
        self.first_enter_return_phase = False
        self.search_steps = 0
        self.search_move_count = 0
        self.is_indoor = False
        self.move_to_the_door_progress = False
        self.need_to_verify_indoor = True
        self.exploration_in_progress
        self.have_explored = set()
        self.space_layout = []
        self.no_door_count = 0

        return 
