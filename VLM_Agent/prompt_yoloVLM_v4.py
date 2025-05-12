
def initial_clue_prompt(clue):
    p = f"""
        You are a rescue robot, and now you have the following initial prompts about the rescue target:
        Clue: {clue}

        Please analyze two major questions:
        1. Does the prompt mention where the house is located relative to the robot? (Front, right side, etc.)
        2. What objects or symbols are mentioned in the prompt? According to the logical description of the prompt, arrange the mentioned landmark objects as nodes in the order from the starting point (away from the injured person) to the ending point (injured person).
        3. What is the room type the injured person lying in? Such as bedroom or warehouse.

        Output format with XML tags:
        <a>Your step-by-step think process </a>
        <b>house's direction (front/back/left/right/left_front/right_front/left_rear/right_rear/none)</b>
        <c>Sorted landmark objects<c>
        <d>room type (warehouse/bedroom/bathroom)</d>

        Important direction notes:
        - For compound directions like "in front of you on your right", use "right_front" 
        - When text contains phrases like "on your right side in front", "to the right and forward", 
          or similar combinations of front/right, use "right_front"
        - Similarly, use "left_front", "right_rear", or "left_rear" for other compound directions
        - Be precise about direction mapping - don't default to simple "front" or "right" when the text 
          clearly indicates a compound direction
        
        Example 1:
        Clue: The injured person is in a bedroom at the end of the corridor on the first floor of a house to your right rear. Please find this person and carry him/her to the yellow ambulance stretcher.
        <a>For the first question, I need to determine the direction of the first target. The clue states the injured person is in a house that is to "your right rear", which means the house is in the right_back direction from the robot's position. For the second question, I need to identify landmark objects mentioned in the clue. The landmarks are: house -> corridor -> bedroom -> injured person. There's also mention of a yellow ambulance stretcher, but this appears to be the destination after finding the person, not a landmark on the way to the person.For the fourth question, the injured person lying in the bedroom.</a>
        <b>right_rear</b>
        <c>house,corridor,bedroom,injured person</c>
        <d>bedroom</d>

        Example 2:
        Clue: There is an injured woman lying in the bedroom on the first floor of the house in front of you. Please find her and carry her to the yellow ambulance stretcher.
        <a>For the first question, I need to determine the direction of the first target. The clue states the injured person is in a house that is "in front of you", which means the house and the injured person inside are in the front direction from the robot's position. For the second question, I need to identify landmark objects mentioned in the clue. The landmarks are: house -> first floor -> bedroom -> injured person. There's also mention of a yellow ambulance stretcher, but this appears to be the destination after finding the person, not a landmark on the way to the person.For the fourth question, the injured person lying in the bedroom.</a>
        <b>front</b>
        <c>house,bedroom,injured woman</c>
        <d>bedroom</d>

    """
    return p

        # Important landmark identification notes:
        # - When the injured person is inside a building (house, warehouse, etc.), include "door" in your landmark list
        #   between the building and interior landmarks
        # - Example: "house,door,corridor,bedroom,injured person" rather than just "house,corridor,bedroom,injured person"
        # - The door is a critical navigational landmark that must be included for indoor rescue scenarios



def whether_in_house_prompt(clue):
    p = f"""

        Analyze the following text: {clue}.
        Determine whether the injured person is inside or outside of the house.

        Use XML tags to output results in the following format:
        <a>Analysis of the clue</a>
        <b>yes/no,if the injured person is inside of the house, output yes, else no</b>

    """
    return p



def search_prompt(landmark_list):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image is a concatenation of three images, reflecting the robot's field of view on the left, front, and right sides, respectively.

        You now have a list of landmarks:[{landmark_str}]
        the list was sort in descending order of priority. 

        Please complete the following tasks:
        1. Please analyze the information and objects contained in each image separately;
        2. Analyze tri-view visual inputs (left/front/right) to determine optimal exploration direction for rescue operations. 

        you can choose your direction referred on following considerations:
        1. **Object Density**: Estimate which section has a higher number of detected objects relative to others. A higher density may indicate an area with more human-related activities or clues.

        2. **Key Interactive Objects**: Check for the presence of important interactive landmarks such as doors, staircases, switches, or hallways. These may lead to new areas or rooms.

        3. **Path Accessibility**: Evaluate if a path is physically navigable — ideally wider than 1.5 meters and free of large obstacles or blockages that would prevent movement.

        4. **Lighting Condition**: Consider the brightness level of each section (0-255). Poorly lit areas may be harder to navigate or less likely to contain useful visual information, while very bright areas may indicate windows, exits, or open space.

        Use XML tags to output results in the following format:
        <a>Analysis of information of each view</a>
        <b> left/right, The direction you choose, you can only choose from left or right</b>

        Example:
        <a>
        Analyzing each view:

        - Left view: The image shows a partially open space with some scattered furniture like a chair and a cabinet. No interactive landmarks such as doors or staircases are visible. Path width appears sufficient, estimated around 2 meters. Brightness level is 180.

        - Front view: Not considered, as only left and right views are eligible for direction selection.

        - Right view: A clearly visible doorway is present with a switch panel next to it. The door appears slightly open, suggesting a new room or area. The path is wide enough (~2.5 meters) and free of major obstacles. Lighting is slightly lower than the left, with brightness around 160.

        The highest-priority landmark in the list is "door", and it appears in the right view.

        </a>
        <b>right</b>

    """
    return p





def search_prompt_begin(landmark_list):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image now displays the visual field in front of the robot.

        You now have a list of landmarks:[{landmark_str}]
        the objects in the list are some iconic objects that may be seen on the path from your current location to the injured person. The objects at the front are of low priority, while the injured at the bottom have the highest priority.

        Please complete the following tasks:
        1. Analyze which objects are present in the field of view;
        2. Corresponding to the iconic objects in the list, find the object with the highest priority in the field of view.
        Please note that the description should be consistent with the description in the list.

        Use XML tags to output results in the following format:
        <a>Analysis of information within the field of view</a>
        <b>The highest priority landmark object (If there is no landmark object in the list within the field of view, output NONE)</b>

        Example:
        <a>In the field of view, I can see the entrance of the orchard, the fence, and a garbage bin. I carefully scan the ground level for any human figure lying down, but don't see an injured person. Among the visible landmarks from my list, the orchard entrance has the highest priority.</a>
        <b>orchard entrance</b>

    """
    return p



def search_prompt_back(landmark_list):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image is a concatenation of four images, reflecting the robot's field of view on the front, right, back and left sides, respectively.

        You need to find the stretcher within your field of view (if it exists). Stretchers will always appear next to the ambulance, so when you cannot see the stretcher, you should output the ambulance.

        Important guidance:
        - Stretchers and ambulances are typically parked on roads or in open areas, not inside buildings or dense vegetation
        - If you cannot see a stretcher or ambulance, look for roads, driveways, or large open areas where emergency vehicles could access
        - Prioritize searching in directions with visible roads, parking lots, or wide open spaces

        Please complete the following tasks:

        1. Please analyze the information and objects contained in each image separately
        2. Check if there is an ambulance or stretcher in the field of view. If there is, output it. When both are visible, prioritize outputting the stretcher and only output one object.
        3. If neither stretcher nor ambulance is visible, identify which direction shows the most road-like area or open space where emergency vehicles could park

        Use XML tags to output results in the following format:

        <a>Check if there is a stretcher or ambulance in the field of view. If none, analyze which direction has the most road-like or open area.</a>
        <b>stretcher/ambulance/road/NONE</b>
        <c>select one from front/right/back/left (which image does the selected object/area belong to)</c>

        Example 1 (when stretcher is visible):
        <a>The left view includes red cars, trash cans, and lawns. There are stretchers and ambulances ahead. There is a swimming pool on the right, with parasols and lounge chairs next to it. The image behind is a wall. The stretcher and ambulance are both within sight, I should prioritize the delivery of the stretcher.</a>
        <b>stretcher</b>
        <c>front</c>

        Example 2 (when no stretcher/ambulance, but road is visible):
        <a>I carefully examine all four views. The front view shows a narrow path between buildings. The right view shows dense vegetation. The back view shows a wall with no passages. The left view shows what appears to be a paved road or driveway where emergency vehicles could access. I cannot see any stretcher or ambulance, so I should report the road area.</a>
        <b>road</b>
        <c>left</c>

        Example 3 (when no clear indication in any direction):
        <a>After examining all four views, I cannot identify any stretcher, ambulance, or clear road access. The front view shows dense trees, the right view shows a building wall, the back view shows more vegetation, and the left view shows rough terrain. None of these directions clearly indicates where an ambulance might be parked.</a>
        <b>NONE</b>
        <c>front</c>
    """
    return p


def move_forward_prompt(target):
    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image now displays the visual field in front of the robot. You are currently moving towards the target :[{target}].
        The image is divided into three parts by two red vertical lines: left, middle, right. Please identify the following issue:
        1. Is the target still within sight?
        2. Is the target mainly located on the left, middle, or right side of the image?

        Important notes on object recognition:
        - Vehicle recognition:
          * When looking for a specific colored vehicle (e.g., "white car", "red car"), be aware that lighting conditions and shadows can significantly affect color perception
          * A white car in shadows might appear gray or even black
          * A red car in low light might appear brown or dark maroon
          * Focus on the vehicle's shape, size and context rather than exact color matching


        Use XML tags to output results in the following format:
        <a>yes/no (Determine if the target is still within the field of view)</a>
        <b>left/middle/right(Determine which area the target is in)</b> 

        Example:
        <a>yes</a>
        <b>middle</b> 

    """
    return p


def search_move_prompt(landmark_list):
    """
    生成用于引导机器人从单幅图像中判断最佳探索方向的提示词，
    路标作为辅助线索参考，核心依据为通向新空间的可能性。
    
    Args:
        landmark_list: 需要寻找的关键路标（参考项）
        
    """
    landmark_str = ", ".join(landmark_list)

    p = f"""
        You are a rescue robot tasked with finding the most promising direction for exploration to locate an injured person.

        The input is a SINGLE image showing the scene directly in front of the robot.
        
        You are using the following landmarks as reference indicators of exploration value: {landmark_str}

        Your goal is to determine whether the image suggests an open path, door, corridor, or other novel space worth exploring.
        Walls, furniture surfaces, or blocked areas are **not** valid exploration targets.

        ### Your Tasks:
        1. Carefully analyze the image to assess whether it contains a structure or passage worth exploring (e.g., doorway, hallway, arch, open room area, corner turn, etc.).
        2. Use the reference landmarks to support your reasoning if any are visible.
        3. If no promising exploration direction is visible (e.g., flat wall, closed cabinet, blocked path), report that.
        4. If exploration is possible, determine where in the image this opportunity is located:
            - Left third → <c>left</c>
            - Middle third → <c>front</c>
            - Right third → <c>right</c>
        
        ### Output Format (use XML):
        <a>Your detailed visual reasoning and judgment of the scene, including mentions of open space, structures, or obstacles.</a>
        <b>Whether this view is worth exploring: "yes" or "no"</b>
        <c>left/front/right/None (best part of the image to explore, or None if fully blocked)</c>

        ### Example 1 (Exploration possible):
        <a>There is a clear open doorway on the left side of the image, showing another room behind it. The lighting difference and shadows confirm spatial depth. The landmark 'wooden door' is visible as well.</a>
        <b>yes</b>
        <c>left</c>

        ### Example 2 (No exploration value):
        <a>The image shows a flat white wall taking up the entire frame. No visible paths, doors, or structural openings. No reference landmarks visible.</a>
        <b>no</b>
        <c>None</c>

        ### Example 3 (Obstructed area):
        <a>The center area appears to lead somewhere, but it is blocked by a table and debris. A door is partially visible behind but inaccessible in the current state.</a>
        <b>no</b>
        <c>front</c>
    """
    return p



def exploration_prompt(explored_doors):
    p = f"""
        You are a rescue robot exploring an unknown indoor environment to locate injured individuals.

        You receive a SINGLE image from your current camera view, which has been visually divided by guiding lines into three parts: **left**, **center**, and **right**.

        Your task is to analyze the image and determine which direction (left, center, or right) is the most promising to explore next, based on the following considerations:

        1. **Object Density**: Estimate which section has a higher number of detected objects relative to others. A higher density may indicate an area with more human-related activities or clues.

        2. **Key Interactive Objects**: Check for the presence of important interactive landmarks such as doors, staircases, switches, or hallways. These may lead to new areas or rooms.

        3. **Path Accessibility**: Evaluate if a path is physically navigable — ideally wider than 1.5 meters and free of large obstacles or blockages that would prevent movement.

        4. **Lighting Condition**: Consider the brightness level of each section (0-255). Poorly lit areas may be harder to navigate or less likely to contain useful visual information, while very bright areas may indicate windows, exits, or open space.

        ---

        Use XML format to provide your reasoning and decision:

        <think>Step-by-step analysis based on object density, key objects, accessibility, and lighting. Explain why you chose a particular direction.</think>  
        <side>left/front/right/none,choose from above choice, If the image is extremely dark or contains no visible content, select none</side>

        Example:
        <think>
        The left section is crowded with furniture, making it less accessible. The right side has poor lighting. The center part contains a door and appears well-lit with no obstacles in the way. It's also the least cluttered area.
        </think>
        <side>front</side>



    """
    return p

def check_person_and_room_type_prompt(room_type_list):
    p = f"""
        tasks:
        1. Is there any injured person lying? Please answer yes or no.
        2. Please confirm the room type in the picture.You can choose from {room_type_list}.

        Use XML tags to output results in the following format:
        <a> yes/no </a>
        <b>room_type</b>
        """
    return p


def position_prompt():
    p = f"""
        
        Please determine the spatial location of the door in the picture, such as the east side of the living room, the end of the corridor, etc. 
        Just return the location description text, keep it as short as possible.

        Select an object around the door as a reference to better identify the different doors.

        Use XML tags to output results in the following format:
        <a> location description </a>
        """
    return p


def is_explored_prompt(current_door, explored_doors):
    p = f"""
        The locations of the doors that have been explored are as follows:{explored_doors}. 
        Please confirm whether the door at the following location: {current_door} have been explored.

        Use XML tags to output results in the following format:
        <a>yes/no, if the door has been explored, output yes, else no</a>

        """
    return p

def room_type_prompt(room_type_list):
    p = f"""
        Please confirm the room type in the picture.
        You can choose from {room_type_list}.

        Use XML tags to output results in the following format:
        <a>room_type</a>

        """
    return p


def move_obstacle_prompt():
    p = f"""
        You are helping a robot move, and the incoming RGB image is its downward view. 
        Please confirm if there are any horizontal obstacles in the view that may hinder our movement, such as a wall, a horizontal fence, etc. And other objects, such as a tree or a box, although they are also obstacles, their width is limited, so we can easily navigate around them, and these are not considered horizontal obstacles.
        In addition, obstacles also need to have sufficient height, and if they are just objects on the ground, they can also be ignored.

        Please output according to the format, including XML tags:
        <a>Whether there are horizontal obstacles</a> (Choose from 0 and 1, where 0 represents none and 1 represents presence)
        <b>If it exists, describe what obstacle it is</b>


        Example:
        <a>1</a>
        <b>There is a white fence in front of me</b>

        """
    return p

def moving_suggestion(clue):
    p = f"""
        You are assisting a robot in locating an injured person lying on the ground. Your task is to provide a **moving direction suggestion** based on the **text clue**, the robot's **current visual observation**, and a **scratch trajectory image** that aids spatial reasoning.

        ## Inputs:
        1. **Text Clue**: {clue}
        2. **Robot’s Observation Sequence**: A concatenation of RGB images representing the robot's recent first-person views.
        3. **Trajectory Image**: A top-down sketch of the robot’s movement history. The **green point** marks the starting location, the **red point** marks the current position, and the **upward direction** represents the robot’s initial orientation. The robot has **no memory** of previously visited paths, so this image is crucial for reasoning.

        ## Your Task:
        Analyze the text clue, image observations, and the trajectory to determine which direction the robot should move in next.

        ## Output Format:
        Provide your suggestion using the following XML format:
        <a>YOUR_SUGGESTION</a>
        (Choose from: front, right, back, left, jump)

        ## Example Output:
        <a>front</a>
    """
    return p



def moving_back_suggestion():
    p = f"""
    You are helping a robot to finding the yellow stretcher beside a ambulance car, your task is to give a moving direction suggestion, considering the robot's observations.

    #Input
    1. Robot's observation sequence: The input RGB image is a concatenation of robot's continuous observation

    Please analyze the robot's observations to determine the direction the robot should explore.
    **Considering Strategy**
    1. First determine if the ambulance car is visible in the current view, if not, turn around to explore.
    2. The ambulance car generally are parking near a large free area, like middle of the road, so avoid moving toward area with clutter structures or buildings.
    
    Please output according to the format, including XML tags:
        <a>The moving suggestion</a> (Select from front, right, back, left)</c>
    
    Example:
    <a>front</a>

    """
    return p

def initial_clue_prompt_v3(clue):
    p = f"""
        You are a rescue robot, and now you have the following initial prompts about the rescue target:
        Clue: {clue}

        Please analyze three major questions:
        1. Does the prompt mention where the first target is located on the robot? (Front, right side, etc.)
        2. What objects or symbols are mentioned in the prompt? According to the logical description of the prompt, arrange the mentioned landmark objects as nodes in the order from the starting point (away from the injured person) to the ending point (injured person).
        3. Does the prompt describe the color of the injured person's clothes? If yes, what color are they?

        Output format with XML tags:
        <a>Your step-by-step think process </a>
        <b>Injured person's direction (front/back/left/right/left_front/right_front/none)</b>
        <c>Sorted landmark objects<c>
        <d>Color of injured person's clothes (if mentioned, otherwise "unknown")</d>

        Important direction notes:
        - For compound directions like "in front of you on your right", use "right_front" 
        - When text contains phrases like "on your right side in front", "to the right and forward", 
          or similar combinations of front/right, use "right_front"
        - Similarly, use "left_front", "right_back", or "left_back" for other compound directions
        - Be precise about direction mapping - don't default to simple "front" or "right" when the text 
          clearly indicates a compound direction

        Example 1:
        Clue: A wounded woman wearing a light floral top and blue jeans is lying in front of you on your right.
        <a> For the first question, I need to determine the direction of the first target. The clue states "in front of you on your right", which indicates a compound direction - specifically right_front direction from the robot's position. For the second question, there are no landmark objects mentioned other than the injured person herself. For the third question, the clothing colors mentioned are "light floral top and blue jeans". </a>
        <b>right_front</b>
        <c>injured person</c>
        <d>light floral top and blue jeans</d>

        Example 2:
        Clue: There is a male injured person wearing a military green shirt and white pants lying in front of your left side. There are some bushes, a mailbox, and a trash can around this man. Please find him and carry him to the yellow ambulance stretcher. The stretcher is in front of you.
        <a>
            For the first question, I need to determine the direction of the first target. The clue clearly states "lying in front of your left side", which indicates a compound direction - specifically left_front direction from the robot's position. 

            For the second question, I need to identify and sort the landmark objects mentioned in the clue. The landmarks mentioned near the injured person are bushes, a mailbox, and a trash can. The clue also mentions a yellow ambulance stretcher that is "in front of you". So the landmarks in order from farthest from the injured person to closest would be: stretcher -> trash can -> mailbox -> bushes -> injured person.

            For the third question, I need to identify the clothing colors mentioned. The injured person is described as "wearing a military green shirt and white pants".
        </a>
        <b>left_front</b>
        <c>stretcher,trash can,mailbox,bushes,injured person</c>
        <d>military green shirt and white pants</d>

    """
    return p