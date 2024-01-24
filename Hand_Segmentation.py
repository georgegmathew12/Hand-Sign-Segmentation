import cv2
from PIL import Image
import io
import numpy as np
import urllib.request
import os
import glob
import mediapipe as mp
import time
import pyautogui
import math
import webbrowser


def get_screenshot_from_frame(frame):
    # Convert the frame to a PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return img

def save_screenshots(url, num_screenshots):
    # Open the video feed using urllib
    stream = urllib.request.urlopen(url)
    bytes = b''

    screenshots = []
    count = 0

    # Generate a unique filename based on timestamp and random number
    timestamp = int(time.time() * 1000)
    random_num = np.random.randint(0, 10000)

    while count < num_screenshots:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            screenshot = get_screenshot_from_frame(frame)
            filename = f"screenshot_{timestamp}_{random_num}_{count}.jpg"  # Update filename to include timestamp and random number
            screenshot.save(filename)
            screenshots.append(screenshot)
            count += 1

    # Close the video feed
    stream.close()

    return screenshots

def delete_all_screenshots(folder_path):
    """
    Deletes all image files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the image files.
    """
    try:
        # Get the list of all files in the folder
        files = os.listdir(folder_path)

        # Loop through all files and delete image files
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file.endswith(".jpg") or file.endswith(".png"):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


    except Exception as e:
        print(f"Failed to delete image files: {e}")

def enhance_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    
    enhanced = clahe.apply(gray)

    return enhanced

def are_fingers_outstretched(hand_landmarks):
    if hand_landmarks is None:
        return False

    thumb_outstretched = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].x

    other_fingers_outstretched = all(
        [
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].y
        ]
    )
    return thumb_outstretched and other_fingers_outstretched

def is_thumbs_up(hand_landmarks):
    if hand_landmarks is None:
        return False
    thumb_outstretched = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].y

    other_fingers_folded = all(
            [   hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].x
            ]
        )
        
    return thumb_outstretched and other_fingers_folded
    
def is_thumbs_down(hand_landmarks):
    if hand_landmarks is None:
        return False
    thumb_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].y
    
    other_fingers_folded = all(
            [        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x <
                    hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].x
            ]
    )
        
    return thumb_folded and other_fingers_folded


def is_longhorn(hand_landmarks):
    if hand_landmarks is None:
        return False

    # Index and pinky fingers are extended
    index_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    pinky_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y
    fingers_extended = (index_tip_y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y) and \
                       (pinky_tip_y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].y)

    # Middle two fingers are folded into the palm
    middle_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
    fingers_folded = (middle_tip_y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y) and \
                     (ring_tip_y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y)

    # Thumb can be extended or folded
    thumb_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x
    thumb_extended = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].y

    # Return True if all conditions are met
    return fingers_extended and fingers_folded and (thumb_folded or thumb_extended)

    
def main():

    # Load the Mediapipe HandPose model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.01)


    #url = 'http://192.168.201.149'
    #url = 'http://192.168.222.149/'
    url = 'http://192.168.143.149'
    
    delete_all_screenshots(r"C:\Users\Evan_\Downloads\hand_images")
    num_screenshots = 7
    screenshots = save_screenshots(url, num_screenshots)
    
    screenshot_to_analyze = screenshots[len(screenshots) // 2]
    frame = np.array(screenshot_to_analyze)
    
    frame_enhanced = enhance_image(frame)
    
    result = hands.process(cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), -1)
            
            if is_thumbs_up(hand_landmarks):
                print("Thumbs up detected!")
                for i in range(50):
                    pyautogui.press('volumeup')
            elif is_thumbs_down(hand_landmarks):
                print("Thumbs down detected!")
                pyautogui.press('volumemute')                
            elif are_fingers_outstretched(hand_landmarks):
                print("All fingers outstretched detected!")
                pyautogui.hotkey('win', 'd')
            elif is_longhorn(hand_landmarks):
                print("BEVO detected!")
                webbrowser.open_new_tab("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
        #frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
        #cv2.imshow(f"Hand {1} with Landmarks", frame_bgr)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        hands.close()
        
        delete_all_screenshots(r'C:\Users\Evan_\Downloads\hand_images')

if __name__ == '__main__':
    while True:
        main()
        time.sleep(1)
