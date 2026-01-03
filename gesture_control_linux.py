"""
Gesture Control System for Linux
Real-time hand gesture recognition for controlling media playback, system navigation, and mouse interactions.
"""

import cv2
import time
import numpy as np
import mediapipe as mp
import subprocess
from dataclasses import dataclass
from typing import Tuple, List, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """System configuration parameters"""
    # Camera settings
    CAM_INDEX = 0
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    DETECTION_MARGIN = 140
    
    # Smoothing factor for mouse movement (0-1)
    MOUSE_SMOOTHING = 0.25
    
    # Pinch detection thresholds (normalized by frame width)
    PINCH_THRESHOLD_DOWN = 0.040
    PINCH_THRESHOLD_UP = 0.055
    
    # Gesture stability requirements
    STABLE_FRAMES_REQUIRED = 8
    COOLDOWN_SECONDS = 0.6
    
    # Swipe detection parameters
    SWIPE_MIN_PIXELS = 170
    SWIPE_MAX_DURATION = 0.45
    
    # Volume control (rock sign)
    VOLUME_STEP_PIXELS = 35
    VOLUME_REPEAT_DELAY = 0.12
    
    # Continuous seeking (two-hand pinch)
    SEEK_MIN_PIXELS = 18
    SEEK_REPEAT_DELAY = 0.06
    
    # Mouse interaction timings
    CLICK_FIRE_WINDOW = 0.12
    DRAG_HOLD_DURATION = 0.25
    
    # Scroll sensitivity
    SCROLL_GAIN = 1.8


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between minimum and maximum"""
    return max(minimum, min(maximum, value))


def euclidean_distance(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points"""
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


def map_to_screen(
    x: int, 
    y: int, 
    frame_width: int, 
    frame_height: int, 
    screen_width: int, 
    screen_height: int
) -> Tuple[float, float]:
    """Map camera coordinates to screen coordinates with margin"""
    x = clamp(x, Config.DETECTION_MARGIN, frame_width - Config.DETECTION_MARGIN)
    y = clamp(y, Config.DETECTION_MARGIN, frame_height - Config.DETECTION_MARGIN)
    
    normalized_x = (x - Config.DETECTION_MARGIN) / (frame_width - 2 * Config.DETECTION_MARGIN)
    normalized_y = (y - Config.DETECTION_MARGIN) / (frame_height - 2 * Config.DETECTION_MARGIN)
    
    return normalized_x * screen_width, normalized_y * screen_height


# =============================================================================
# SYSTEM CONTROL (Linux - xdotool)
# =============================================================================

def send_key(key: str) -> None:
    """Send a key press using xdotool"""
    try:
        subprocess.run(["xdotool", "key", key], check=False, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("ERROR: xdotool not found. Install with: sudo apt install xdotool")


def open_spotify() -> None:
    """Focus existing Spotify window or launch it"""
    try:
        result = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--class", "Spotify"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        if result.stdout.strip():
            window_id = result.stdout.strip().split("\n")[0]
            subprocess.run(["xdotool", "windowactivate", window_id])
            return
    except Exception:
        pass
    
    # Launch Spotify if not running
    subprocess.Popen(["spotify"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =============================================================================
# HAND GESTURE RECOGNITION
# =============================================================================

def get_fingers_up(landmarks: List[Tuple[int, int]]) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Determine which fingers are extended.
    Returns: (thumb, index, middle, ring, pinky)
    
    Landmark indices:
    - Thumb: tip=4, ip=3
    - Index: tip=8, pip=6
    - Middle: tip=12, pip=10
    - Ring: tip=16, pip=14
    - Pinky: tip=20, pip=18
    """
    thumb = landmarks[4][0] > landmarks[3][0]  # Horizontal comparison for thumb
    index = landmarks[8][1] < landmarks[6][1]
    middle = landmarks[12][1] < landmarks[10][1]
    ring = landmarks[16][1] < landmarks[14][1]
    pinky = landmarks[20][1] < landmarks[18][1]
    
    return thumb, index, middle, ring, pinky


def is_ok_sign(landmarks: List[Tuple[int, int]], frame_width: int) -> bool:
    """OK sign: thumb+index pinched, other fingers up"""
    pinch_dist = euclidean_distance(landmarks[4], landmarks[8]) / float(frame_width)
    _, _, middle, ring, pinky = get_fingers_up(landmarks)
    return (pinch_dist < 0.04) and middle and ring and pinky


def is_ily_sign(landmarks: List[Tuple[int, int]], frame_width: int) -> bool:
    """I Love You sign: thumb, index, pinky up; middle+ring down"""
    thumb, index, middle, ring, pinky = get_fingers_up(landmarks)
    return thumb and index and pinky and (not middle) and (not ring)


def is_rock_sign(landmarks: List[Tuple[int, int]]) -> bool:
    """Rock sign: index+pinky up, middle+ring down"""
    _, index, middle, ring, pinky = get_fingers_up(landmarks)
    return index and pinky and (not middle) and (not ring)


def is_three_finger(landmarks: List[Tuple[int, int]]) -> bool:
    """Three fingers: index+middle+ring up, pinky down"""
    _, index, middle, ring, pinky = get_fingers_up(landmarks)
    return index and middle and ring and (not pinky)


def is_point_only(landmarks: List[Tuple[int, int]]) -> bool:
    """Pointing: only index up"""
    _, index, middle, ring, pinky = get_fingers_up(landmarks)
    return index and (not middle) and (not ring) and (not pinky)


def is_fist(landmarks: List[Tuple[int, int]]) -> bool:
    """Fist: all fingers down"""
    _, index, middle, ring, pinky = get_fingers_up(landmarks)
    return (not index) and (not middle) and (not ring) and (not pinky)


def is_open_palm(landmarks: List[Tuple[int, int]]) -> bool:
    """Open palm: all non-thumb fingers up"""
    _, index, middle, ring, pinky = get_fingers_up(landmarks)
    return index and middle and ring and pinky


def is_call_me_sign(landmarks: List[Tuple[int, int]]) -> bool:
    """Call me sign: thumb+pinky up, others down"""
    thumb, index, middle, ring, pinky = get_fingers_up(landmarks)
    return thumb and pinky and (not index) and (not middle) and (not ring)


def is_pinch_down(landmarks: List[Tuple[int, int]], frame_width: int) -> bool:
    """Check if thumb and index are pinched together"""
    return (euclidean_distance(landmarks[4], landmarks[8]) / float(frame_width)) < Config.PINCH_THRESHOLD_DOWN


def is_pinch_up(landmarks: List[Tuple[int, int]], frame_width: int) -> bool:
    """Check if thumb and index are separated"""
    return (euclidean_distance(landmarks[4], landmarks[8]) / float(frame_width)) > Config.PINCH_THRESHOLD_UP


# =============================================================================
# GESTURE STABILITY TRACKER
# =============================================================================

@dataclass
class GestureStabilityTracker:
    """Track gesture stability over multiple frames"""
    gesture_name: str = "NONE"
    frame_count: int = 0
    
    def update(self, new_gesture: str) -> None:
        """Update with new gesture detection"""
        if new_gesture == self.gesture_name:
            self.frame_count += 1
        else:
            self.gesture_name = new_gesture
            self.frame_count = 1
    
    def is_stable(self) -> bool:
        """Check if gesture has been stable for required frames"""
        return self.frame_count >= Config.STABLE_FRAMES_REQUIRED


# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def main():
    """Main gesture control loop"""
    import pyautogui
    
    # Configure PyAutoGUI
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    
    # Initialize camera
    cap = cv2.VideoCapture(Config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
    
    screen_width, screen_height = pyautogui.size()
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # State variables
    prev_mouse_pos = None
    last_frame_time = time.time()
    
    # Mouse interaction state
    pinch_state = "up"
    pinch_down_timestamp = None
    is_dragging = False
    
    # Scroll state
    last_scroll_y = None
    
    # Cooldown tracking
    last_action_time = {}
    
    def can_execute_action(action_key: str) -> bool:
        """Check if enough time has passed since last action"""
        current_time = time.time()
        return (action_key not in last_action_time) or \
               ((current_time - last_action_time[action_key]) > Config.COOLDOWN_SECONDS)
    
    def mark_action_executed(action_key: str) -> None:
        """Mark that an action was just executed"""
        last_action_time[action_key] = time.time()
    
    # Swipe tracking
    swipe_start = None  # (x_position, timestamp)
    swipe_gesture_type = None
    
    # Volume control state
    last_volume_y = None
    last_volume_time = 0.0
    
    # Two-hand seeking state
    last_seek_time = 0.0
    
    # Gesture stability tracker
    stable_gesture = GestureStabilityTracker()
    
    # Initialize hand detection
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        
        print("Gesture Control Active - Press 'q' to quit")
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / max(1e-6, (current_time - last_frame_time))
            last_frame_time = current_time
            
            # Draw detection boundary
            cv2.rectangle(
                frame, 
                (Config.DETECTION_MARGIN, Config.DETECTION_MARGIN),
                (width - Config.DETECTION_MARGIN, height - Config.DETECTION_MARGIN),
                (255, 255, 255), 
                2
            )
            
            # HUD messages
            hud_messages = []
            
            # Process detected hands
            detected_hands = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmark coordinates
                    landmarks = []
                    for i in range(21):
                        x = int(hand_landmarks.landmark[i].x * width)
                        y = int(hand_landmarks.landmark[i].y * height)
                        landmarks.append((x, y))
                    detected_hands.append((landmarks, hand_landmarks))
                    
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
            
            # =================================================================
            # TWO-HAND GESTURES: Continuous Seeking
            # =================================================================
            if len(detected_hands) == 2:
                landmarks_1, _ = detected_hands[0]
                landmarks_2, _ = detected_hands[1]
                
                both_pinched = is_pinch_down(landmarks_1, width) and is_pinch_down(landmarks_2, width)
                
                if both_pinched:
                    # Use average of index finger tips
                    avg_x = (landmarks_1[8][0] + landmarks_2[8][0]) / 2.0
                    
                    # Track horizontal movement
                    if "seek_last_x" not in last_action_time:
                        last_action_time["seek_last_x"] = avg_x
                    
                    delta_x = avg_x - last_action_time["seek_last_x"]
                    last_action_time["seek_last_x"] = avg_x
                    
                    # Execute seek if movement threshold met
                    if abs(delta_x) > Config.SEEK_MIN_PIXELS and \
                       (current_time - last_seek_time) > Config.SEEK_REPEAT_DELAY:
                        last_seek_time = current_time
                        
                        if delta_x > 0:
                            send_key("XF86AudioForward")
                            hud_messages.append("SEEK: Forward")
                        else:
                            send_key("XF86AudioRewind")
                            hud_messages.append("SEEK: Rewind")
            
            # =================================================================
            # SINGLE-HAND GESTURES
            # =================================================================
            if len(detected_hands) >= 1:
                landmarks, _ = detected_hands[0]
                
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                
                # -------------------------------------------------------------
                # Mouse Movement (pointing gesture)
                # -------------------------------------------------------------
                if is_point_only(landmarks):
                    screen_x, screen_y = map_to_screen(
                        index_tip[0], index_tip[1],
                        width, height,
                        screen_width, screen_height
                    )
                    
                    if prev_mouse_pos is None:
                        prev_mouse_pos = (screen_x, screen_y)
                    
                    # Apply smoothing
                    smooth_x = prev_mouse_pos[0] + (screen_x - prev_mouse_pos[0]) * Config.MOUSE_SMOOTHING
                    smooth_y = prev_mouse_pos[1] + (screen_y - prev_mouse_pos[1]) * Config.MOUSE_SMOOTHING
                    prev_mouse_pos = (smooth_x, smooth_y)
                    
                    pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
                
                # -------------------------------------------------------------
                # Mouse Click/Drag State Machine
                # -------------------------------------------------------------
                pinch_distance = euclidean_distance(thumb_tip, index_tip) / float(width)
                
                # Update pinch state
                if pinch_state == "up" and pinch_distance < Config.PINCH_THRESHOLD_DOWN:
                    pinch_state = "down"
                    pinch_down_timestamp = current_time
                elif pinch_state == "down" and pinch_distance > Config.PINCH_THRESHOLD_UP:
                    pinch_state = "up"
                    pinch_down_timestamp = None
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                
                # Right click: pinch + middle finger up
                thumb, index, middle, ring, pinky = get_fingers_up(landmarks)
                if pinch_state == "down" and middle and index and not is_dragging:
                    if pinch_down_timestamp and \
                       (current_time - pinch_down_timestamp) < Config.CLICK_FIRE_WINDOW and \
                       can_execute_action("right_click"):
                        pyautogui.click(button="right")
                        mark_action_executed("right_click")
                        pinch_down_timestamp = None
                        hud_messages.append("MOUSE: Right Click")
                
                # Left click: pinch down moment
                if pinch_state == "down" and not is_dragging:
                    if pinch_down_timestamp and \
                       (current_time - pinch_down_timestamp) < Config.CLICK_FIRE_WINDOW and \
                       can_execute_action("left_click"):
                        pyautogui.click(button="left")
                        mark_action_executed("left_click")
                        pinch_down_timestamp = None
                        hud_messages.append("MOUSE: Left Click")
                
                # Drag: hold pinch
                if pinch_state == "down" and pinch_down_timestamp and \
                   (current_time - pinch_down_timestamp) > Config.DRAG_HOLD_DURATION and \
                   not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                    hud_messages.append("MOUSE: Dragging")
                
                # -------------------------------------------------------------
                # Scrolling (index + middle up, no pinch)
                # -------------------------------------------------------------
                if index and middle and (not ring) and (not pinky) and pinch_state == "up":
                    current_y = (index_tip[1] + middle_tip[1]) / 2.0
                    
                    if last_scroll_y is None:
                        last_scroll_y = current_y
                    
                    delta_y = current_y - last_scroll_y
                    last_scroll_y = current_y
                    
                    scroll_amount = int(-delta_y * Config.SCROLL_GAIN)
                    if abs(scroll_amount) > 0:
                        pyautogui.scroll(scroll_amount)
                        hud_messages.append("MOUSE: Scrolling")
                else:
                    last_scroll_y = None
                
                # -------------------------------------------------------------
                # Discrete Media/System Gestures (with stability)
                # -------------------------------------------------------------
                current_gesture = "NONE"
                
                if is_fist(landmarks):
                    current_gesture = "PLAY_PAUSE"
                elif is_ok_sign(landmarks, width):
                    current_gesture = "MUTE"
                elif is_ily_sign(landmarks, width):
                    current_gesture = "FULLSCREEN"
                elif is_call_me_sign(landmarks):
                    current_gesture = "OPEN_SPOTIFY"
                
                stable_gesture.update(current_gesture)
                
                if stable_gesture.is_stable():
                    if stable_gesture.gesture_name == "PLAY_PAUSE" and can_execute_action("play_pause"):
                        send_key("XF86AudioPlay")
                        mark_action_executed("play_pause")
                        hud_messages.append("MEDIA: Play/Pause")
                    
                    elif stable_gesture.gesture_name == "MUTE" and can_execute_action("mute"):
                        send_key("XF86AudioMute")
                        mark_action_executed("mute")
                        hud_messages.append("MEDIA: Mute")
                    
                    elif stable_gesture.gesture_name == "FULLSCREEN" and can_execute_action("fullscreen"):
                        send_key("f")
                        mark_action_executed("fullscreen")
                        hud_messages.append("MEDIA: Fullscreen")
                    
                    elif stable_gesture.gesture_name == "OPEN_SPOTIFY" and can_execute_action("spotify"):
                        open_spotify()
                        mark_action_executed("spotify")
                        hud_messages.append("SYS: Open Spotify")
                
                # -------------------------------------------------------------
                # Volume Control (Rock Sign - continuous)
                # -------------------------------------------------------------
                if is_rock_sign(landmarks):
                    current_y = (landmarks[8][1] + landmarks[20][1]) / 2.0
                    
                    if last_volume_y is None:
                        last_volume_y = current_y
                    
                    delta_y = current_y - last_volume_y
                    
                    if abs(delta_y) > Config.VOLUME_STEP_PIXELS and \
                       (current_time - last_volume_time) > Config.VOLUME_REPEAT_DELAY:
                        last_volume_time = current_time
                        steps = int(abs(delta_y) // Config.VOLUME_STEP_PIXELS)
                        
                        if delta_y < 0:  # Moving up
                            for _ in range(steps):
                                send_key("XF86AudioRaiseVolume")
                            hud_messages.append("MEDIA: Volume Up")
                        else:  # Moving down
                            for _ in range(steps):
                                send_key("XF86AudioLowerVolume")
                            hud_messages.append("MEDIA: Volume Down")
                        
                        last_volume_y = current_y
                else:
                    last_volume_y = None
                
                # -------------------------------------------------------------
                # Swipe Gestures (Track Next/Prev, Window Switching)
                # -------------------------------------------------------------
                current_swipe_type = None
                anchor_x = None
                
                if is_point_only(landmarks):
                    current_swipe_type = "POINT"
                    anchor_x = index_tip[0]
                elif is_three_finger(landmarks):
                    current_swipe_type = "THREE"
                    anchor_x = (landmarks[8][0] + landmarks[12][0] + landmarks[16][0]) / 3.0
                
                if current_swipe_type is not None:
                    if swipe_start is None or swipe_gesture_type != current_swipe_type:
                        swipe_start = (anchor_x, current_time)
                        swipe_gesture_type = current_swipe_type
                    else:
                        start_x, start_time = swipe_start
                        elapsed_time = current_time - start_time
                        delta_x = anchor_x - start_x
                        
                        if elapsed_time <= Config.SWIPE_MAX_DURATION and \
                           abs(delta_x) >= Config.SWIPE_MIN_PIXELS:
                            
                            if swipe_gesture_type == "POINT":
                                # Media track navigation
                                if delta_x > 0 and can_execute_action("next_track"):
                                    send_key("XF86AudioNext")
                                    mark_action_executed("next_track")
                                    hud_messages.append("MEDIA: Next Track")
                                elif delta_x < 0 and can_execute_action("prev_track"):
                                    send_key("XF86AudioPrev")
                                    mark_action_executed("prev_track")
                                    hud_messages.append("MEDIA: Previous Track")
                            
                            elif swipe_gesture_type == "THREE":
                                # Window switching
                                if delta_x > 0 and can_execute_action("alt_tab"):
                                    send_key("alt+Tab")
                                    mark_action_executed("alt_tab")
                                    hud_messages.append("SYS: Next Window")
                                elif delta_x < 0 and can_execute_action("alt_shift_tab"):
                                    send_key("shift+alt+Tab")
                                    mark_action_executed("alt_shift_tab")
                                    hud_messages.append("SYS: Previous Window")
                            
                            swipe_start = None
                            swipe_gesture_type = None
                        
                        # Reset if too slow
                        if elapsed_time > Config.SWIPE_MAX_DURATION:
                            swipe_start = (anchor_x, current_time)
                else:
                    swipe_start = None
                    swipe_gesture_type = None
            
            else:
                # No hands detected - reset states
                prev_mouse_pos = None
                last_scroll_y = None
                stable_gesture.update("NONE")
            
            # =================================================================
            # RENDER HUD
            # =================================================================
            # FPS counter
            cv2.putText(
                frame, 
                f"FPS: {fps:.1f}", 
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Action messages
            y_position = 35
            for message in hud_messages[:5]:
                cv2.putText(
                    frame, 
                    message, 
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                y_position += 30
            
            # Display frame
            cv2.imshow("Gesture Control (Press 'q' to quit)", frame)
            
            # Check for quit
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture Control Terminated")


if __name__ == "__main__":
    main()
