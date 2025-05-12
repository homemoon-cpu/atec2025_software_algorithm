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
from ultralytics import YOLO


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

        self.explored_doors = []        # 已探索的门列表
        self.current_door_target = None # 当前目标门信息
        self.door_move_threshold = 50   # 判断是否为同一门的像素距离阈值
        self.door_width_threshold_outhouse = 0.3 # 门宽占屏幕宽度的开门阈值
        self.door_width_threshold_inhouse = 0.4
        self.room_type_list = ['warehouse','bedroom','bathroom']

        self.last_action_was_open = False
        self.roomtype = 'unknown'

        self.door_boxes = []
        self.out_door_box = None
        self.person_boxes = []

        self.names = ['out_door','person','ambulance','strecher','door']

        self.is_in_house = True

        self.handle.write("finish yoloVLM agent initialize\n")
        self.handle.flush()


    def predict(self, obs, info, boxes):

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
            if self.is_in_house:
                return self._handle_initial_phase()
            else:
                return self._handle_initial_phase_v3()
            
        elif self.phase == 1:
            if self.is_in_house:
                return self._handle_search_phase(boxes)
            else:
                return self._handle_search_phase_v3()
            
        elif self.phase == 2:
            # Rescue the injured person
            return self._handle_rescue_phase()
            
        elif self.phase == 3:
            if self.is_in_house:
                return self._handle_return_phase_v4(boxes)
            else:
                return self._handle_return_phase_v3()
            
        elif self.phase == 4:
            # Place the person on the stretcher
            return self._handle_placement_phase()
        
        # Default action if nothing else to do
        return ([0, 0], 0, 0)
    
    def analyse_whether_in_house(self):
        prompt = whether_in_house_prompt(self.clue)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_llm(prompt=prompt)
                self.handle.write(res)
                self.handle.flush()
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    is_in_house = match.group(2).strip()
                    if is_in_house:
                        print("is_in_house:",is_in_house)
                        return True
                    else:
                        print("is_in_house:",is_in_house)
                        return False
    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[CLUE ANALYSE] Failed after {max_retries} attempts: {str(e)}")

    def _handle_initial_phase(self):
        # First time through, analyze the clue
        if not self.landmark:
            self.analyse_initial_clue()
            self.handle.write(f"LANDMARK LIST: {', '.join(self.landmark)}\n")
            self.handle.flush()
            self.phase = 1
            return ([0,0],0,0)
        
        # Return default action if already initialized
        return ([0, 0], 0, 0)
    
        
    def _handle_search_phase(self,boxes):
        self.door_boxes = []
        # 门检测处理
        for box in boxes:
            cls = int(box.cls.item())
            if(self.names[cls] == 'out_door' or self.names[cls] == 'door') :
                self.out_door_box = (box.xywh[0].tolist())
            if (self.names[cls] == 'door') and self.is_indoor:
                self.door_boxes.append(box.xywh[0].tolist())
            if (self.names[cls] == 'person') and self.is_indoor:
                self.person_boxes.append(box.xywh[0].tolist())

        print("need_to_verify_indoor:",self.need_to_verify_indoor)
        if self.need_to_verify_indoor:
            if self._verify_door_entry():
                self.need_to_verify_indoor = False
                self.exploration_in_progress = True
            else:
                self.exploration_in_progress = False
                return self.move_to_the_door_outhouse(self.out_door_box)
            
        if self.info['picked'] == 1:
            self.phase = 3
            self.search_in_progress = True
            self.move_to_the_door_progress = False
            self.exploration_in_progress = False
            self.landmark = ['ambulance', 'stretcher']

        # YOLO检测
        # if self.info['picked'] == 1:
        #     results = self.yolo_model(source=self.obs, imgsz=640, conf=0.1)
        # else:base_speed
        if (not self.move_to_the_door_progress) and len(self.door_boxes) > 0:
            current_obs = self.obs.copy()
            self.handle.write("identify unexplored door\n")
            self.handle.flush()
            print("identify unexplored door")
            # 获取空间位置描述
            prompt = position_prompt()
            res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
            pattern = re.compile(r'<a>(.*?)</a>', re.DOTALL)
            match = pattern.search(res)
            if match:
                position_desc = match.group(1).strip()
                self.handle.write(f"current detect door's position_desc: {position_desc}\n")
                self.handle.flush()
                print("current detect door's position_desc:", position_desc)
            
            # 检查是否已探索
            position_desc_expolred = [d['position_desc'] for d in self.explored_doors]
            prompt = is_explored_prompt(position_desc, position_desc_expolred)
            res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
            pattern = re.compile(r'<a>(.*?)</a>', re.DOTALL)
            match = pattern.search(res)
            if match:
                is_explored = match.group(1).strip()
                self.handle.write(f"is_explored:{is_explored}\n")
                self.handle.flush()
                print(f"is_explored:{is_explored}")
            
            if is_explored == 'no':
                # 创建门记录
                self.current_door_target = {
                    'position_desc': position_desc,
                    'box': self.door_boxes[0],  # 取第一个检测到的门
                    'preview_image': current_obs
                }
                self.move_to_the_door_progress = True
                self.exploration_in_progress = False
            elif is_explored == 'yes':
                self.exploration_in_progress = True

        if self.exploration_in_progress:
            self.handle.write(f"exploration\n")
            self.handle.flush()
            print("exploration")
            has_direction = self._process_exploration_result()
            if has_direction:
                self.handle.write(f"has direction\n")
                self.handle.flush()
                print('has direction')
                self.exploration_in_progress = True
                return ([0,0],0,0)
            else:
                self.handle.write(f"no direction, trapped\n")
                self.handle.flush()
                print('no direction, trapped')
                self.exploration_in_progress = False
                self.search_in_progress = True
                return self._start_search()
            
        elif self.move_to_the_door_progress:
            self.handle.write(f"move_to_the_door\n")
            self.handle.flush()
            print('move_to_the_door')
            if not self.last_action_was_open:
                action =  self.move_to_the_door_inhouse()
                if action[2] == 6:
                    self.exploration_in_progress = True
                    self.move_to_the_door_progress = False
                    return ([0,-20],0,0)
                # print("self.last_action_was_open:",self.last_action_was_open)
                if action[2] == 5:
                    self.handle.write(f"open the door\n")
                    self.handle.flush()
                    print("open the door")
                    self.action_buffer = []
                    self.action_buffer.append(([0, 100], 0, 0))
                    self.obs_buffer.append(False)
                    self.action_buffer.append(([0, 100], 0, 0))
                    self.obs_buffer.append(False)
                    self.last_action_was_open = True
                    return action

            if self.last_action_was_open:
                self.last_action_was_open = False
                vlm_check = self._check_room_type()
                if len(self.person_boxes)>0:
                    return ([0,0],0,0) #由yolo完成救援任务
                self.handle.write(f"self.roomtype: {self.roomtype}\n")
                self.handle.flush()
                self.handle.write(f"vlm_check['room_type']:{vlm_check['room_type']}\n")
                self.handle.flush()
                print("self.roomtype:",self.roomtype)
                print("vlm_check['room_type']:",vlm_check['room_type'])
                if self.roomtype == vlm_check['room_type']:
                    self.handle.write('enter the room and exploration\n')
                    self.handle.flush()
                    print('enter the room and exploration')
                    self.current_door_target['room_type'] = vlm_check['room_type']
                    self.explored_doors.append(self.current_door_target)
                    self.exploration_in_progress = True
                    self.move_to_the_door_progress = False
                    self.current_door_target = None
                    return ([0,50],0,0)
                else:
                    self.handle.write('not target roomtype,turn back and exploration\n')
                    self.handle.flush()
                    print('not target roomtype,turn back and exploration')
                    self.current_door_target['room_type'] = vlm_check['room_type']
                    self.explored_doors.append(self.current_door_target)
                    self.exploration_in_progress = True
                    self.move_to_the_door_progress = False
                    self.current_door_target = None
                    self.action_buffer = []
                    for _ in range(5):
                        self.action_buffer.append(([-30, 0], 0, 0))
                        self.obs_buffer.append(False)
                    self.action_buffer.append(([0, 0], 0, 5))
                    self.obs_buffer.append(False)
                    self.action_buffer.append(([0, 100], 0, 0))
                    self.obs_buffer.append(False)
                    return ([-30, 0], 0, 0)
                
            else:
                return action
        
        elif self.search_in_progress:
            self.phase = 1
            self.handle.write('search\n')
            self.handle.flush()
            print('search')
            self._process_search_result()
            self.search_in_progress = False
            self.exploration_in_progress = True
            return ([0,0],0,0)

        else:
            self.search_in_progress = True
            return self._start_search()
    
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
        
    def _check_room_type(self):
        prompt = room_type_prompt(self.room_type_list)
        res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
        pattern = re.compile(r'<a>(.*?)</a>', re.DOTALL)
        match = pattern.search(res)
        if match:
            room_type = match.group(1).strip()
        
        return {
            'room_type': room_type
        }
    
    def move_to_the_door_outhouse(self,out_door_box):
        img = self.obs
        img_h, img_w = img.shape[:2]
        x_center, y_center, width, height = out_door_box

        # 计算相对参数
        horizontal_offset = x_center - img_w/2
        width_ratio = width / img_w
        vertical_ratio = y_center / img_h

        # 速度映射函数
        def map_speed(ratio, max_speed=100):
            return int(max(-100, min(100, ratio * max_speed)))

        def map_steer(ratio, max_steer=30):
            return int(max(-30, min(30, ratio * max_steer)))

        # 运动控制逻辑
        if width_ratio < self.door_width_threshold_outhouse:
            if abs(horizontal_offset) > img_w * 0.15:
                # 动态转向控制
                steer_ratio = (horizontal_offset / img_w) * 2.0  # [-1,1]
                steer = map_steer(steer_ratio)
                
                # 垂直位置影响前进速度
                base_speed = 80 if vertical_ratio > 0.7 else 60
                return ([steer, 0], 0, 0)
            else:
                # 渐进加速：离门越近速度越慢
                speed = map_speed(0.8 - (width_ratio/self.door_width_threshold_outhouse)*0.5)
                return ([0, speed], 0, 0)
        
        else:
            self.need_to_verify_indoor = True
            self.action_buffer.append(([0, 100], 0, 0))
            self.obs_buffer.append(False)
            self.action_buffer.append(([0, 100], 0, 0))
            self.obs_buffer.append(False)
            self.action_buffer.append(([0, 0], 0, 0))  
            self.obs_buffer.append(True) 
            return ([0, 0], 0, 5)  # 开门动作
        
    def move_to_the_door_inhouse(self):
        img = self.obs
        img_h, img_w = img.shape[:2]
        if len(self.door_boxes) == 0:
            return ([0, 0], 0, 6)
        box = self.door_boxes[0]
        x_center, y_center, width, height = box

        # 计算相对参数
        horizontal_offset = x_center - img_w/2
        width_ratio = width / img_w
        vertical_ratio = y_center / img_h

        # 速度映射函数
        def map_speed(ratio, max_speed=100):
            return int(max(-100, min(100, ratio * max_speed)))

        def map_steer(ratio, max_steer=30):
            return int(max(-30, min(30, ratio * max_steer)))

        # 运动控制逻辑
        if width_ratio < self.door_width_threshold_inhouse:
            if abs(horizontal_offset) > img_w * 0.15:
                # 动态转向控制
                steer_ratio = (horizontal_offset / img_w) * 2.0  # [-1,1]
                steer = map_steer(steer_ratio)
                
                # 垂直位置影响前进速度
                base_speed =70 if vertical_ratio > 0.7 else 50
                return ([steer, 0], 0, 0)
            else:
                # 渐进加速：离门越近速度越慢
                speed = map_speed(0.8 - (width_ratio/self.door_width_threshold_inhouse)*0.5)
                return ([0, speed], 0, 0)

        else:
            return ([0, 0], 0, 5)  # 开门动作


    def _process_exploration_result(self):
        prompt = exploration_prompt(self.explored_doors)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.add_visual_guides(self.obs)))
                self.handle.write(f"[EXPLORATION] \n {res} \n\n\n")
                self.handle.flush()
                print(f"[EXPLORATION] \n {res} \n\n\n")
                pattern = re.compile(r'<think>(.*?)</think>\s*<side>(.*?)</side>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    side = match.group(2).strip()
                    if side == "left":
                        self.handle.write('left\n')
                        self.handle.flush()
                        print("left")
                        self.action_buffer.append(([-30, 0], 0, 0))
                        self.obs_buffer.append(False)
                        self.action_buffer.append(([0, 100], 0, 0))
                        self.obs_buffer.append(False)
                        return True 
                    elif side == "right":
                        self.handle.write('right\n')
                        self.handle.flush()
                        print("right")
                        self.action_buffer.append(([30, 0], 0, 0))
                        self.obs_buffer.append(False)
                        self.action_buffer.append(([0, 100], 0, 0))
                        self.obs_buffer.append(False)
                        return True 
                    elif side == "front":
                        self.handle.write('front\n')
                        self.handle.flush()
                        print("front")
                        self.action_buffer.append(([0, 100], 0, 0))
                        self.obs_buffer.append(False)
                        return True 
                    elif side == "center":
                        self.handle.write('front\n')
                        self.handle.flush()
                        print("front")
                        self.action_buffer.append(([0, 100], 0, 0))
                        self.obs_buffer.append(False)
                        return True 
                    elif side == "None":
                        self.handle.write('None\n')
                        self.handle.flush()
                        print('None')
                        return False
                else:
                    return 

                
            except Exception as e:
                self.handle.write(f"Error in exploration: {str(e)}\n")
                self.handle.flush()
    
    def _handle_rescue_phase(self):
        # If person is picked, move to next phase
        if self.info['picked'] == 1:
            self.phase = 3
            self.handle.write("[SPEAK] I have rescued them. It's time to go back.\n")
            self.handle.flush()
            print("[SPEAK] I have rescued them. It's time to go back.\n")
            self.search_in_progress = True
            self.move_to_the_door_progress = False
            self.exploration_in_progress = False
            self.landmark = ['ambulance', 'stretcher']
        
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
    
    # def _handle_return_phase_v4(self,boxes):
    #     # Update landmarks for the return journey
    #     self.handle.write('enter return phase\n')
    #     self.handle.flush()
    #     print("enter return phase\n")
    #     for box in boxes:
    #         cls = int(box.cls.item())
    #         if(self.names[cls] == 'out_door' or self.names[cls] == 'door') :
    #             self.out_door_box = (box.xywh[0].tolist())
    #         if (self.names[cls] == 'door') and self.is_indoor:
    #             self.door_boxes.append(box.xywh[0].tolist())
        
    #     if not self.last_action_was_open and 
        

    #     if self.first_enter_return_phase == False:
    #         self.first_enter_return_phase = True
    #         if (not self.move_to_the_door_progress) and len(self.door_boxes) > 0:
        
    #     if self.search_in_progress:
    #         result = self._process_search_result()
    #         self.search_in_progress = False
            
    #         if self.target == "stretcher" or self.target == "Stretcher":
    #             self.phase = 4
    #             self.handle.write("[SPEAK] I found the stretcher. Let me place the injured person on it.\n")
    #             self.handle.flush()
    #             print("[SPEAK] I found the stretcher. Let me place the injured person on it.\n")
    #             return ([0, 50], 0, 0)
                
    #         if result == True:  # landmark exists in view
    #             if self.move_obstacle_in_progress:
    #                 obstacle_result = self._process_obstacle_result()
    #                 if obstacle_result == True:
    #                     self.move_obstacle_in_progress = False
    #                     return self._start_search()
                
    #             self.move_to_landmark_in_progress = True
    #             return self._start_move_to_landmark()
    #         else:
    #             self.search_in_progress = True
    #             return self._start_search()
                
        
    #     # Start the search process
    #     else:
    #         self.search_in_progress = True
    #         return self._start_search()

    def _handle_return_phase_v4(self):
        # Update landmarks for the return journey
        self.handle.write('enter return phase\n')
        self.handle.flush()
        print("enter return phase\n")
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
        

    def _handle_return_phase_v3(self):
        # Update landmarks for the return journey
        self.handle.write('enter return phase\n')
        self.handle.flush()
        print("enter return phase\n")
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

        # 添加左右三分线
        left_third = w // 3
        right_third = w * 2 // 3
        cv2.line(marked_image, (left_third, 0), (left_third, h), (0, 0, 255), 1)
        cv2.line(marked_image, (right_third, 0), (right_third, h), (0, 0, 255), 1)
        
        return marked_image
    
    
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
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>\s*<d>(.*?)</d>', re.DOTALL)
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

                    self.roomtype = match.group(4).strip()
                     
                    return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[CLUE ANALYSE] Failed after {max_retries} attempts: {str(e)}")
        

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
        self.last_action_was_open = False
        self.roomtype = 'unknown'
        self.door_boxes = []
        self.out_door_box = None
        self.person_boxes = []
        self.is_in_house = self.analyse_whether_in_house()

        return 
    
    def _handle_initial_phase_v3(self):
        # First time through, analyze the clue
        if not self.landmark:
            self.analyse_initial_clue_v3()
            self.handle.write(f"LANDMARK LIST: {', '.join(self.landmark)}\n")
            # Directly search in the initial direction without rotating
            prompt = search_prompt_begin(self.landmark, self.person_text)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    res = call_api_vlm(prompt=prompt, base64_image=self.encode_image_array(self.obs))
                    self.handle.write(f"[SEARCH] \n {res} \n\n\n")
                    pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>', re.DOTALL)
                    match = pattern.search(res)

                    if match:
                        analysis = match.group(1).strip()
                        landmark = match.group(2).strip()

                        if landmark != "None" and landmark != "NONE":
                            self.target = landmark
                            self.phase = 1
                            self.move_to_landmark_in_progress = True
                            # Queue up the initial forward movement
                            self.action_buffer.append(([0, 100], 0, 1))
                            self.action_buffer.append(([0, 100], 0, 1))
                            self.action_buffer.append(([0, 100], 0, 1))
                            return ([0, 0], 0, 0)
                        else:
                            self.phase = 1
                            self.search_move_in_progress = True
                            self.start_search_from_initial = True
                            return self._start_search_move()

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"[SEARCH] Failed after {max_retries} attempts: {str(e)}")
            
            # If we get here, no landmark was found
            self.phase = 1
            self.search_in_progress = True
            return self._start_search()
        
        # Return default action if already initialized
        return ([0, 0], 0, 0)
    

    def analyse_initial_clue_v3(self):
        prompt = initial_clue_prompt(self.clue)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = call_api_llm(prompt=prompt)
                self.handle.write(res)
                # 修改正则表达式以匹配新增的<d>标签
                pattern = re.compile(r'<a>(.*?)</a>\s*<b>(.*?)</b>\s*<c>(.*?)</c>\s*<d>(.*?)</d>', re.DOTALL)
                match = pattern.search(res)

                if match:
                    direction = match.group(2).strip()
                    self.initial_direction = direction
                    # direction = self.extract_direction(self.clue,['man', 'person', 'injured'])
                    # Queue appropriate rotation actions based on direction
                    if direction == "left":
                        for _ in range(3):
                            self.action_buffer.append(([-30, 0], 0, 0))
                    elif direction == "right":
                        for _ in range(3):
                            self.action_buffer.append(([30, 0], 0, 0))
                    elif direction == "back":
                        for _ in range(6):
                            self.action_buffer.append(([30, 0], 0, 0))
                    elif "front" in direction and "left" in direction:
                        self.action_buffer.append(([-30, 0], 0, 0))
                        self.action_buffer.append(([-15, 0], 0, 0))
                    elif "front" in direction and "right" in direction:
                        self.action_buffer.append(([30, 0], 0, 0))
                        self.action_buffer.append(([15, 0], 0, 0))

                    # Parse landmarks
                    self.landmark = match.group(3).strip().split(",")
                    self.landmark = ["injured person" if landmark.strip().lower() == "injured woman" or landmark.strip().lower() == "injured man"
                                else landmark.strip() for landmark in self.landmark]
                    self.landmark = [landmark for landmark in self.landmark if "stretcher" not in landmark and "ambulance" not in landmark]
                    self.landmark_back = ['ambulance','stretcher']
                    
                    # 处理伤员衣物颜色信息
                    clothes_color = match.group(4).strip()
                    if clothes_color and clothes_color.lower() != "unknown":
                        self.person_text = clothes_color
                        self.handle.write(f"[INFO] Detected injured person's clothes color: {self.person_text}\n")
                    
                    return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"[CLUE ANALYSE] Failed after {max_retries} attempts: {str(e)}")
                

    def _handle_search_phase_v3(self):
        if self.info['picked'] == 1:
            self.phase = 3
        # If search is currently in progress, continue it
        if self.search_in_progress:
            result = self._process_search_result()
            self.search_in_progress = False
            
            if result == True:  # landmark exists
                self.move_to_landmark_in_progress = True
                self.move_steps = 0  # Reset step counter when starting new movement
                return self._start_move_to_landmark()
            else:
                self.search_move_in_progress = True
                return self._start_search_move()
                
        # If moving to landmark is in progress
        elif self.move_to_landmark_in_progress:
            result = self._process_move_result()
            
            if result == False:
                self.move_fail_count += 1
                self.move_to_landmark_in_progress = False
                self.search_move_in_progress = True
                return self._start_search_move()
                
            # Check if we've reached max steps for move_to_landmark
            self.move_steps += 1
            if self.move_steps >= self.max_move_steps:
                # If reached max steps, go back to search
                self.move_to_landmark_in_progress = False
                self.search_move_in_progress = True
                return self._start_search_move()
                
            if self.target == "injured person" or self.target == "Injured person":
                self.phase = 2
                self.move_to_landmark_in_progress = False
                self.handle.write("[SPEAK] I have found the injured person! Now I will try to rescue them.\n")
                
                # Queue up two forward actions
                self.action_buffer.append(([0, 50], 0, 3))
                self.obs_buffer.append(False)
                return ([0, 50], 0, 0)
            
            # Continue moving to landmark
            return self._start_move_to_landmark()
        
        elif self.search_move_in_progress:
            result = self._process_search_move_result()
            self.search_move_in_progress = False
            
            if result == True:
                self.move_to_landmark_in_progress = True  
                return self._start_move_to_landmark()
            else:
                self.search_move_in_progress = True
                return self._start_search_move()
            
        # If neither search nor move is in progress, start search
        else:
            self.search_in_progress = True
            return self._start_search()
