
def initial_clue_prompt(clue):
    p = f"""
        You are a rescue robot, and now you have the following initial prompts about the rescue target:
        Clue: {clue}

        Please analyze two major questions:
        1. Does the prompt mention where the house is located relative to the robot? (Front, right side, etc.)
        2. What objects or symbols are mentioned in the prompt? According to the logical description of the prompt, arrange the mentioned landmark objects as nodes in the order from the starting point (away from the injured person) to the ending point (injured person).

        Output format with XML tags:
        <a>Your step-by-step think process </a>
        <b>house's direction (front/back/left/right/left_front/right_front/left_rear/right_rear/none)</b>
        <c>Sorted landmark objects<c>

        Important direction notes:
        - For compound directions like "in front of you on your right", use "right_front" 
        - When text contains phrases like "on your right side in front", "to the right and forward", 
          or similar combinations of front/right, use "right_front"
        - Similarly, use "left_front", "right_rear", or "left_rear" for other compound directions
        - Be precise about direction mapping - don't default to simple "front" or "right" when the text 
          clearly indicates a compound direction
        
        Example 1:
        Clue: The injured person is in a bedroom at the end of the corridor on the first floor of a house to your right rear. Please find this person and carry him/her to the yellow ambulance stretcher.
        <a>For the first question, I need to determine the direction of the first target. The clue states the injured person is in a house that is to "your right rear", which means the house is in the right_back direction from the robot's position. For the second question, I need to identify landmark objects mentioned in the clue. The landmarks are: house -> corridor -> bedroom -> injured person. There's also mention of a yellow ambulance stretcher, but this appears to be the destination after finding the person, not a landmark on the way to the person.</a>
        <b>right_rear</b>
        <c>house,corridor,bedroom,injured person</c>

        Example 2:
        Clue: There is an injured woman lying in the bedroom on the first floor of the house in front of you. Please find her and carry her to the yellow ambulance stretcher.
        <a>For the first question, I need to determine the direction of the first target. The clue states the injured person is in a house that is "in front of you", which means the house and the injured person inside are in the front direction from the robot's position. For the second question, I need to identify landmark objects mentioned in the clue. The landmarks are: house -> first floor -> bedroom -> injured person. There's also mention of a yellow ambulance stretcher, but this appears to be the destination after finding the person, not a landmark on the way to the person.</a>
        <b>front</b>
        <c>house,bedroom,injured woman</c>

    """
    return p

        # Important landmark identification notes:
        # - When the injured person is inside a building (house, warehouse, etc.), include "door" in your landmark list
        #   between the building and interior landmarks
        # - Example: "house,door,corridor,bedroom,injured person" rather than just "house,corridor,bedroom,injured person"
        # - The door is a critical navigational landmark that must be included for indoor rescue scenarios


def initial_clue_prompt_indoor(clue):
    p = f"""
        You are a rescue robot, and now you have the following initial prompts about the rescue target:
        Clue: {clue}

        Please analyze two major questions:
        1. Does the prompt mention where the first target is located on the robot? (Front, right side, etc.)
        2. What objects or symbols are mentioned in the prompt? According to the logical description of the prompt, arrange the mentioned landmark objects as nodes in the order from the starting point (away from the injured person) to the ending point (injured person).
        - Attention, you already know that the injured person is indoors, so the node must include a "door".
        Output format with XML tags:
        <a> Your step-by-step think process </a>
        <b>first target's direction (front/back/left/right/none)</b>
        <c>Sorted landmark objects<c>

        Example:
        Clue: The injured person is under a fruit tree. After the fruit tree enters the orchard, turn left. There is a fence outside the orchard and a car is parked.
        <a> (think process) </a>
        <b>front</b>
        <c>car,fence,orchard entrance,fruit tree,injured person</c>


    """
    return p

def in_out_door_prompt(clue):
    p = f"""
    Here are clue about the target:
    {clue}
    and you will get the image of the target.
    Please determine whether the target is indoors or outdoors based on this text. 
    If indoors, please return 0; If outdoors, please return to 1. 
    Do not provide any additional responses.

    """
    return p

def initial_image_prompt():
    p = f"""
    Please analyze the reference image showing an injured person lying on the ground.
    
    Focus only on determining the color of the clothes worn by the injured person.
    
    If there are multiple injured people or if you're unsure, provide the most distinctive color combination you can see.
    
    Please only output colors and do not include any other explanatory text.
    
    Example 1 (correct):
    Blue and white
    
    Example 2 (incorrect, as only color output is allowed):
    The color of the clothes passed on by this person is blue and white
    
    Example 3 (correct):
    Red with black stripes
    """
    return p

def door_prompt():
    p = f"""
        You are a rescue robot, you need to find and approach the door.
        The input RGB image now displays the visual field in front of the robot.
        
            ### Core Task:
        1. Door detection (yes/no)
        2. If door exists:
        a. Position analysis:
            - Left third → <position>left</position>
            - Central third → <position>middle</position>
            - Right third → <position>right</position>
        b. Proximity/Operability check (yes/no):
            * Critical indicators for immediate operation:
                • Door occupies >30% of image width (close enough)
                • Clear visibility of operable components (handle/access panel)
                • No physical obstructions in operating zone
        3. If no door:
        a. No structural elements → <position>NONE</position><operable>NONE</operable>
        b. Extreme close-up (door fills frame showing only edges/frame) 
            → <position>TOO_CLOSE</position><operable>yes</operable> (already in contact position)

        Be careful not to mistake cabinets for doors.

        Use the following visual cues to identify a cabinet (not a door):

        Cabinets often protrude from the wall, while doors are flush with the wall surface.

        Only detect a door if it is flush with the wall, reaches floor level, and has a full door-frame.

        ### Output Format:
        <a>Technical assessment - explicitly note door coverage percentage, component visibility, and spatial relationships</a>
        <position>left/middle/right/NONE/TOO_CLOSE</position>
        <operable>yes/no/NONE</operable>

        ### Enhanced Examples:
        Example 1 (Operable):
        <a>Door occupies 30% width with centered handle (12cm visible clearance). No obstructions in 1m operational radius.</a>
        <position>middle</position>
        <operable>yes</operable>

        Example 2 ():
        <a>Door at 30% width - reach proximity threshold. Handle partially visible about 30cm front.</a>
        <position>right</position>
        <operable>yes</operable>

        Example 3 (Contact position):
        <a>Image shows only door edge textures at 5cm range - confirm physical contact with door surface.</a>
        <position>TOO_CLOSE</position>
        <operable>yes</operable>
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
        2.Analyze tri-view visual inputs (left/front/right) to determine optimal exploration direction for rescue operations. 
search_move_prompt
        Analyzing each view:
        - Left view: Contains a visible corridor with open space ahead. No door or stairs detected.
        - Front view: Contains a large, clearly visible door suggesting a new room. The door width is estimated over 70cm, indicating viable passage.
        - Right view: Shows a cluttered area with furniture, and a narrow path less than 50cm. No visible landmarks.
        Based on the priority list ["door", "corridor", "stairs"], the front image contains the highest-priority landmark: "door".
        Therefore, the object check confirms that the highest-priority item, "door", appears in the front view.
        </a>
        <b>front</b>

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



def exploration_prompt(have_explored, landmark_list):
    p = f"""
        You are a rescue robot, you need to explore the environment and find the wounded.

        The input is a SINGLE image from your current camera view. 

        You have explored the following places: {have_explored}.

        Tasks:
        1.Carefully analyze the image and Confirm where in the house the perspective is observed, such as the living room or bedroom.
        2.Analysis of interior space layout, for example, there is a door on the left side of the living room and a corridor on the right side of the living room.
        3.Check if there are any targets worth exploring, such as doors, corridors, or places that may point to another space.
          The landmark_list is {landmark_list},you can choose your target based on this.
          For each target worth exploring, give the location of the target worth exploring relative to the current location, choose from north/south/west/earst.
          Assume the direction you face when you enter the living room is north.
        4.For target, determine its position in the image: left side, center, or right side

        If a injured person is detected, the target is set to 'injured person'.
        
        The format of target is the target name join with a underline join with the location of the target worth exploring relative to the current location joint with the current location.
        For example, if your target is the door on the west side of your current position, such as living room, then the format of target is "door_west_living room"

        Use XML tags to output results in the following format:
        <think> Your analysis process </think>
        <space_layout> space layout of the house </space_layout>
        <current_position> current postion of yourself, such as the living room, warahouse or bedroom/</current_position>
        <target> target worth exploring, such as door or corridor ,if no target, output None</target>
        <side> left/front/right (which part of the image contains the target), or "None" if no target visible</side>

        Example1:
        <think>
        The image shows a space with a bed, a bedside cabinet, and a window with curtains. These elements are typical of a bedroom. 
        On the right side of the image, there appears to be a wooden door slightly ajar, suggesting it may lead to another space. 
        Given the presence of a door and the layout of the furniture, it's reasonable to conclude that this is a bedroom with a door on the east side.
        The landmark_list includes "door", which matches the visible structure. This door might lead to a corridor or another room, making it worth exploring.
        </think>
        <space_layout>The bedroom has a door on the east side, a bed in the center, and windows possibly facing north.</space_layout>
        <current_position>bedroom</current_position>
        <target>door_east_bedroom</target>
        <side>right</side>

        Example2:
        <think>
        The image shows a spacious area with a sofa, coffee table, TV stand, and a large carpet, which are indicative of a living room. 
        To the left side of the image, there is an open doorway without a door, revealing a narrow, dimly-lit path that appears to be a corridor. 
        This corridor is likely to lead to other unexplored rooms, such as bedrooms or storage areas. 
        Since the direction we face when entering the living room is considered north, the corridor is located to the west of the current position.
        The landmark_list includes "corridor", which aligns with the visual observation.
        </think>
        <space_layout>The living room has a sofa and TV in the center, a corridor to the west, and an open view to the north side.</space_layout>
        <current_position>living room</current_position>
        <target>corridor_west_living room</target>
        <side>left</side>


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

def compare_reference_image():
    p = f"""
        You are a rescue robot analyzing a current view to determine if it leads toward where an injured person was spotted.
        
        The image is structured as follows:
        - TOP: Reference image showing where an injured person was located
        - BOTTOM: Your current view
        
        CRITICAL TASK: Analyze whether your current view shows a path that would likely lead toward the location in the reference image.
        
        FOCUS PRIMARILY ON THESE ENVIRONMENTAL FEATURES SURROUNDING THE INJURED PERSON IN THE REFERENCE IMAGE:
        1. Distinctive terrain/ground features - grass type, soil color, ground texture, slopes
        2. Nearby objects - trees, bushes, rocks, furniture, structures
        3. Background elements - buildings, walls, fences, landscape features
        4. Lighting conditions and shadows - similar light direction, intensity
        5. Spatial arrangement of surrounding elements
        
        For the current view, assess:
        * Which part of the view (left, center, right) shows terrain most similar to the reference image
        * Whether key landmark objects from the reference appear in a particular part of the view
        * If the spatial layout suggests a path toward the scene in the reference
        * Whether the overall environmental context is similar
        
        Use XML tags to structure your response:
        <analysis>Detailed comparison between current view and reference image, focusing on surrounding environmental features</analysis>
        <best_direction>left/front/right (which part of the current view appears most promising for leading toward the reference scene)</best_direction>
        <confidence>Score from 1-10 indicating how confident you are in this assessment</confidence>
        <reasoning>Clear explanation focused on environmental features around the injured person that led to your direction choice</reasoning>
        
        EXAMPLES OF STRONG REASONING:
        
        Example 1: "The right third of your current view shows the same distinctive red-brown soil and scattered white rocks visible around the injured person in the reference image. I can also see similar pine trees with the same distinctive branching pattern that appears in the background of the reference image."
        
        Example 2: "The left part of your current view contains the same type of concrete barriers with yellow markings that are visible near the injured person in the reference image. The lighting angle and shadow patterns are also consistent, suggesting this direction leads toward the reference scene."
        
        Example 3: "The central portion of your view shows asphalt with the same distinctive crack pattern visible in the reference image. The arrangement of trees along the path matches the reference scene's background, and both images share the same mountainous horizon line."
        """
    return p

# def compare_reference_image():
#     p = f"""
#         You are a rescue robot analyzing a panoramic view.
        
#         The image shows three views stitched together:
#         - LEFT THIRD: Your left view
#         - MIDDLE THIRD: Your front view
#         - RIGHT THIRD: Your right view
        
#         Compare these views with the reference scene where an injured person was located.
#         Determine which direction (left, front, or right) shows the most similar environment 
#         to the reference scene and would most likely lead you toward the injured person.
        
#         Focus on:
#         1. Matching landmarks or objects
#         2. Similar terrain or structural elements
#         3. Spatial layout and environmental context
#         4. Visual cues suggesting a path toward the target scene
        
#         Use XML tags to structure your response:
#         <analysis>Detailed comparison of all three views with reference to the target scene</analysis>
#         <best_direction>left/front/right (which direction appears most promising)</best_direction>
#         <confidence>Score from 1-10 indicating how confident you are in this assessment</confidence>
#         <reasoning>Clear explanation of why you selected this direction</reasoning>
#         """
#     return p