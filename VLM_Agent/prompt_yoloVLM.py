
def initial_clue_prompt(clue):
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


def search_prompt(landmark_list, person_text):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image is a concatenation of three images, reflecting the robot's field of view on the left, front, and right sides, respectively.

        You now have a list of landmarks:[{landmark_str}]
        the list was sort in descending order of priority. 

        Please complete the following tasks:
        1. Please analyze the information and objects contained in each image separately;
        2. Check if the objects in the list appear within the field of view.

        If your target is an Injured person, please note that:
        * They are human beings LYING ON THE GROUND, not standing or sitting
        
        Important notes on object recognition:
        - Vehicle recognition:
          * When looking for a specific colored vehicle (e.g., "white car", "red car"), be aware that lighting conditions and shadows can significantly affect color perception
          * A white car in shadows might appear gray or even black
          * A red car in low light might appear brown or dark maroon

        - Ground-level scanning for injured persons:
          * Scan the ground carefully, especially in open areas, under trees, or near landmarks
          * An injured person will appear as a human figure lying horizontally on the ground
          * They might be easily confused with shadows, logs, or other elongated objects
          * Look for color patterns matching the described clothing ({person_text})

        Use XML tags to output results in the following format:
        <a>Check whether the objects in the list appear in order based on the information within the field of view</a>
        <b>the object in the views with the highest priority, or outputs NONE to indicate that all objects in the list are invisible</b> (When describing an object, please ensure consistency with the expression in the list)
        <c> choose one from left/front/right (Which image does the chosen object belong to) </c>

        Example:
        <a>The left view includes red car, trash can, and lawn. There is a wall in front, which should be the wall of the house. There is a swimming pool on the right, with parasols and lounge chairs next to it. I carefully scan the ground in all three views for any human figure lying down, but don't see an injured person. In order of priority, the red car exists in the field of view and belongs to the left view.</a>
        <b>red car</b>
        <c>left</c>

    """
    return p





def search_prompt_begin(landmark_list, person_text):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image now displays the visual field in front of the robot.

        You now have a list of landmarks:[{landmark_str}]
        the objects in the list are some iconic objects that may be seen on the path from your current location to the injured person. The objects at the front are of low priority, while the injured at the bottom have the highest priority.

        If your target is an Injured person, please note that:
        * They are human beings LYING ON THE GROUND, not standing or sitting
        * The color of their clothes is {person_text}
        * Look carefully at ground level for human figures in horizontal position
        * They might appear as elongated shapes on the ground with the specified clothing color
        * They might be partially hidden by grass, debris, or terrain irregularities
        * Pay special attention to any human-shaped objects at ground level that don't match the vertical orientation of standing people

        Important notes on object recognition:
        - Vehicle recognition:
          * When looking for a specific colored vehicle (e.g., "white car", "red car"), be aware that lighting conditions and shadows can significantly affect color perception
          * A white car in shadows might appear gray or even black
          * A red car in low light might appear brown or dark maroon
          * Focus on the vehicle's shape, size and context rather than exact color matching
        
        - Ground-level scanning for injured persons:
          * Scan the ground carefully, especially in open areas, under trees, or near landmarks
          * An injured person will appear as a human figure lying horizontally on the ground
          * They might be easily confused with shadows, logs, or other elongated objects
          * Look for color patterns matching the described clothing ({person_text})
          * Even partial visibility (just legs or torso) should be reported if it matches the description

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


def move_forward_prompt(target, person_text):
    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image now displays the visual field in front of the robot. You are currently moving towards the target :[{target}].
        The image is divided into three parts by two red vertical lines: left, middle, right. Please identify the following issue:
        1. Is the target still within sight?
        2. Is the target mainly located on the left, middle, or right side of the image?

        If your target is an Injured person, please note that:
        * They are human beings LYING ON THE GROUND, not standing or sitting
        * The color of their clothes is {person_text}
        * Look carefully at ground level for human figures in horizontal position
        * They might appear as elongated shapes on the ground with the specified clothing color
        * Do not confuse with standing people - the injured person will be in a prone position

        Important notes on object recognition:
        - Vehicle recognition:
          * When looking for a specific colored vehicle (e.g., "white car", "red car"), be aware that lighting conditions and shadows can significantly affect color perception
          * A white car in shadows might appear gray or even black
          * A red car in low light might appear brown or dark maroon
          * Focus on the vehicle's shape, size and context rather than exact color matching

        - Ground-level scanning for injured persons:
          * Scan the ground carefully for human figures in horizontal position
          * Look for color patterns matching the described clothing ({person_text})
          * Check areas that might contain a person-sized object lying flat

        Use XML tags to output results in the following format:
        <a>yes/no (Determine if the target is still within the field of view)</a>
        <b>left/middle/right(Determine which area the target is in)</b> 

        Example:
        <a>yes</a>
        <b>middle</b> 

    """
    return p
# def search_move_prompt(landmark_list, person_text):
#     landmark_str = ", ".join(landmark_list)

#     p = f"""
#         You are a rescue robot tasked with finding landmarks that will lead you to an injured person.
        
#         The person you need to find is described as: {person_text}
        
#         You are looking for the following landmarks to guide you: {landmark_str}
        
#         The input RGB image is a concatenation of three images, reflecting the robot's field of view on the left, front, and right sides, respectively.
        
#         IMPORTANT COLOR PERCEPTION GUIDELINES:
#         1. Be aware that shadows and lighting conditions can significantly alter the apparent colors of objects
#         2. A white or light-colored object in shadow may appear darker
#         3. When identifying vehicles, focus on shape and context first, then consider color
#         4. Compare relative brightness and hue across the image rather than relying on absolute color values
#         5. For objects in shadow areas, mentally compensate for the lighting effect before determining color
        
#         Tasks:
#         1. Carefully analyze the image for any of the target landmarks
#         2. If you identify a landmark, name it precisely as it appears in the landmark list
#         3. If no landmarks are visible, respond with "None"
#         4. If the same landmark appears in multiple views (left/front/right), choose the view where the landmark appears more centrally positioned in the frame
        
#         Use XML tags to structure your response:
#         <a>Your step-by-step analysis of visible landmarks or explanation of why none are visible</a>
#         <b>Name of identified landmark exactly as it appears in the landmark list, or "None" if no landmarks are visible</b>
#         <c>left/front/right (which view contains the identified landmark in the most central position), else respond with "None"</c>
        
#         Example 1:
#         <a>I can see a white car parked under a tree. The car is in shadow, which makes it appear gray-dark, but it is clearly a white car based on its overall brightness compared to truly dark objects in the scene. The white car appears in both the front and right views, but it is more centrally positioned in the front view.</a>
#         <b>white car</b>
#         <c>front</c>
        
#         Example 2:
#         <a>I've carefully examined all three views but cannot identify any of the specific landmarks listed. I can see some trees and a path, but none of the key landmarks mentioned in the list are visible.</a>
#         <b>None</b>
#         <c>None</c>
        
#         Example 4:
#         <a>I can see what appears to be a red car in both the left and right views. Despite the shadows making color identification difficult, I can tell it's a red car based on its distinctive hue that is clearly reddish-brown rather than white or silver, even accounting for shadow effects. The car in the right view is more centrally positioned and appears larger, suggesting it's closer and would be a better navigation target.</a>
#         <b>red car</b>
#         <c>right</c>
#     """
#     return p

def search_move_prompt(landmark_list):
    """
    生成用于当前视图分析的提示词，适用于单一观察图像而非三联图
    
    Args:
        landmark_list: 需要寻找的路标列表
        
    Returns:
        提示词字符串
    """
    landmark_str = ", ".join(landmark_list)

    p = f"""
        You are a rescue robot tasked with finding landmarks that will lead you to an injured person.
        
        You are looking for the following landmarks to guide you: {landmark_str}
        
        The input is a SINGLE image from your current camera view.
        
        IMPORTANT COLOR PERCEPTION GUIDELINES:
        1. Be aware that shadows and lighting conditions can significantly alter the apparent colors of objects
        2. A white or light-colored object in shadow may appear darker, grayish
        3. A red object in shadow may appear darker, brownish, or even blackish
        4. When identifying vehicles, focus on shape and context first, then consider color
        5. Compare relative brightness and hue across the image rather than relying on absolute color values
        6. For objects in shadow areas, mentally compensate for the lighting effect before determining color
        
        CRITICAL GUIDANCE FOR DISTINGUISHING REAL INJURED PERSONS FROM SHADOWS:
        1. Shadows lack physical texture and detail - they appear as flat dark areas on the ground
        2. Real people, even from a distance, show some physical dimension and volume
        3. Shadows typically have uniform darkness, while real people have variations in tone and color
        4. Shadows have sharp, often distorted outlines, while real people have natural body contours
        5. The edges of shadows are typically cleaner and more defined than the boundaries of real objects
        6. Real injured persons will show some clothing color and texture details that shadows lack
        
        Tasks:
        1. Carefully analyze the image for any of the target landmarks
        2. If you identify a landmark, name it precisely as it appears in the landmark list
        3. If no landmarks are visible, respond with "None"
        4. For landmarks, determine their position in the image: left side, center, or right side
        
        VISUAL POSITIONING GUIDE:
        - LEFT: Landmark appears in the left third of the image
        - CENTER/FRONT: Landmark appears in the middle third of the image
        - RIGHT: Landmark appears in the right third of the image
        
        Use XML tags to structure your response:
        <a>Your step-by-step analysis of visible landmarks or explanation of why none are visible</a>
        <b>Name of identified landmark exactly as it appears in the landmark list, or "None" if no landmarks are visible</b>
        <c>left/front/right (which part of the image contains the identified landmark), or "None" if no landmark visible</c>
        
        Example 1:
        <a>I can see a white car parked under a tree in the right portion of the image. The car is in shadow, which makes it appear slightly gray-blue, but it is clearly a white car based on its overall brightness compared to truly dark objects in the scene.</a>
        <b>white car</b>
        <c>right</c>
        
        Example 2:
        <a>I've carefully examined the image but cannot identify any of the specific landmarks listed. I can see some trees and a path, but none of the key landmarks mentioned in the list are visible.</a>
        <b>None</b>
        <c>None</c>
        
        Example 3:
        <a>I notice what initially appears to be a person lying on the ground in the left portion of the image. However, upon closer examination, I can tell this is actually a shadow cast by the nearby tree. It lacks physical volume, has uniformly dark coloration without any clothing details, has unnaturally sharp edges, and perfectly conforms to the ground texture. It's clearly just a shadow pattern and not an injured person.</a>
        <b>None</b>
        <c>None</c>
        
        Example 4:
        <a>I can see an injured person lying on the ground near the center of the image. Unlike a shadow, they have visible physical volume, variations in tone and color consistent with clothing and skin, and natural body contours rather than sharp shadow edges.</a>
        <b>injured person</b>
        <c>front</c>
    """
    return p

# def search_move_prompt(landmark_list):
#     landmark_str = ", ".join(landmark_list)

#     p = f"""
#         You are a rescue robot tasked with finding landmarks that will lead you to an injured person.
        
#         You are looking for the following landmarks to guide you: {landmark_str}
        
#         The input RGB image is a concatenation of three images, reflecting the robot's field of view on the left, front, and right sides, respectively.
        
#         IMPORTANT COLOR PERCEPTION GUIDELINES:
#         1. Be aware that shadows and lighting conditions can significantly alter the apparent colors of objects
#         2. A white or light-colored object in shadow may appear darker
#         3. When identifying vehicles, focus on shape and context first, then consider color
#         4. Compare relative brightness and hue across the image rather than relying on absolute color values
#         5. For objects in shadow areas, mentally compensate for the lighting effect before determining color
        
#         CRITICAL GUIDANCE FOR DISTINGUISHING REAL INJURED PERSONS FROM SHADOWS:
#         1. Shadows lack physical texture and detail - they appear as flat dark areas on the ground
#         2. Shadows typically have uniform darkness, while real people have variations in tone and color
       

#         Tasks:
#         1. Carefully analyze the image for any of the target landmarks
#         2. If you identify a landmark, name it precisely as it appears in the landmark list
#         3. If no landmarks are visible, respond with "None"
#         4. If the same landmark appears in multiple views (left/front/right), choose the view where the landmark appears more centrally positioned in the frame
        
#         Use XML tags to structure your response:
#         <a>Your step-by-step analysis of visible landmarks or explanation of why none are visible</a>
#         <b>Name of identified landmark exactly as it appears in the landmark list, or "None" if no landmarks are visible</b>
#         <c>left/front/right (which view contains the identified landmark in the most central position), else respond with "None"</c>
        
#         Example 1:
#         <a>I can see a white car parked under a tree. The car is in shadow, which makes it appear gray-dark, but it is clearly a white car based on its overall brightness compared to truly dark objects in the scene. The white car appears in both the front and right views, but it is more centrally positioned in the front view.</a>
#         <b>white car</b>
#         <c>front</c>
        
#         Example 2:
#         <a>I've carefully examined all three views but cannot identify any of the specific landmarks listed. I can see some trees and a path, but none of the key landmarks mentioned in the list are visible.</a>
#         <b>None</b>
#         <c>None</c>
        
#         Example 4:
#         <a>I can see what appears to be a red car in both the left and right views. Despite the shadows making color identification difficult, I can tell it's a red car based on its distinctive hue that is clearly reddish-brown rather than white or silver, even accounting for shadow effects. The car in the right view is more centrally positioned and appears larger, suggesting it's closer and would be a better navigation target.</a>
#         <b>red car</b>
#         <c>right</c>
#     """
#     return p

# def search_move_prompt(landmark_list, person_text):

#     landmark_str = ", ".join(landmark_list)

#     p = f"""

#         You are a rescue robot, you need to find the wounded.

#         The input RGB image is a concatenation of three images, reflecting the robot's field of view on the left, front, and right sides, respectively.

#         You now have a list of landmarks:[{landmark_str}]
#         the list was sort in descending order of priority. 

#         Please complete the following tasks:
#         1. Please analyze the information and objects contained in each image separately;
#         2. Check if the objects in the list appear within the field of view.

#         If your target is an Injured person, please note that:
#         * They are human beings LYING ON THE GROUND, not standing or sitting
        
#         Important notes on object recognition:
#         - Vehicle recognition:
#           * When looking for a specific colored vehicle (e.g., "white car", "red car"), be aware that lighting conditions and shadows can significantly affect color perception
#           * A white car in shadows might appear gray or even black

#         - Ground-level scanning for injured persons:
#           * Scan the ground carefully, especially in open areas, under trees, or near landmarks
#           * An injured person will appear as a human figure lying horizontally on the ground
#           * They might be easily confused with shadows, logs, or other elongated objects
#           * Look for color patterns matching the described clothing ({person_text})

#         Use XML tags to output results in the following format:
#         <a>Check whether the objects in the list appear in order based on the information within the field of view</a>
#         <b>the object in the views with the highest priority, or outputs NONE to indicate that all objects in the list are invisible</b> (When describing an object, please ensure consistency with the expression in the list)
       
#         Example:
#         <a>The left view includes red car, trash can, and lawn. There is a wall in front, which should be the wall of the house. There is a swimming pool on the right, with parasols and lounge chairs next to it. I carefully scan the ground in all three views for any human figure lying down, but don't see an injured person. In order of priority, the red car exists in the field of view and belongs to the left view.</a>
#         <b>red car</b>


#     """
#     return p

def exploration_prompt():
    p = f"""

        You are a rescue robot, you need to explore the environment and find the wounded.
        The input RGB image now is a concatenation of three images, showing the visual content of the robot's left, front, and right sides respectively.
        Please analyze the field of view information within each of the three images and choose the direction that is more worth exploring.
        How to determine if it is worth exploring:
        If there are distant extensions or new areas such as corners or channels connecting in the picture, it is more worth exploring; If it's a wall or a corner, then it's obviously not worth exploring.
        Use XML tags to output results in the following format:
        <think> Your analysis process </think>
        <output>The direction you choose can be left, front, or right</output>

        Example:
        <think> In front of me is a wall, and on the right is an empty space, which seems to have no exploratory value. There is a door on my left, and the space inside the door is an unknown area worth exploring. </think>
        <output>left</output>
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