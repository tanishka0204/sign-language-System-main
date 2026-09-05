import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import copy
import itertools
import pyttsx3
import threading
import time
from collections import deque, Counter

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

# --- Helper Functions (Copied/Adapted from app.py to avoid import execution) ---
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def draw_landmarks_on_white(image_size, landmark_point):
    # Create a white image
    white_img = np.ones(image_size, dtype=np.uint8) * 255
    
    if len(landmark_point) > 0:
        # Define connections
        connections = [
            (2, 3), (3, 4),               # Thumb
            (5, 6), (6, 7), (7, 8),       # Index
            (9, 10), (10, 11), (11, 12),  # Middle
            (13, 14), (14, 15), (15, 16), # Ring
            (17, 18), (18, 19), (19, 20), # Little
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0) # Palm
        ]
        
        # Draw lines (Green for skeleton)
        for p1, p2 in connections:
             cv.line(white_img, tuple(landmark_point[p1]), tuple(landmark_point[p2]), (0, 255, 0), 4)

        # Draw keypoints (Red circles)
        for index, landmark in enumerate(landmark_point):
            cv.circle(white_img, (landmark[0], landmark[1]), 5, (0, 0, 255), -1)

    return white_img

# --- Main Application Class ---
class ASLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language To Text Conversion")
        self.geometry("1400x800")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Configuration
        self.cap_device = 0
        self.cap_width = 640  # Reduced for UI fit
        self.cap_height = 480
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.5
        
        # State
        self.current_sentence = ""
        self.last_char = ""
        
        # Smoothing & Timer State
        self.history_length = 15
        self.prediction_history = deque(maxlen=self.history_length)
        self.last_stable_char = ""
        self.stable_char_start_time = 0
        self.char_added_flag = False
        self.auto_add_threshold = 5.0 # Seconds
        
        # Suggestions Data
        self.common_words = [
            "HELLO", "HELP", "HERE", "HOME", "HOW", "HAPPY", 
            "YES", "YOU", "YOUR", "YEAR", 
            "NO", "NOT", "NOW", "NAME", "NICE", 
            "THANK", "THAT", "THIS", "THEY", "TIME",
            "PLEASE", "PEOPLE", "PLAY",
            "GOOD", "GREAT", "GO",
            "WHAT", "WHERE", "WHEN", "WHY", "WHO",
            "I", "IS", "IN", "IT",
            "MY", "ME", "MORE",
            "A", "AND", "ARE", "ABOUT", "ALL"
        ]
        self.suggestion_buttons = []
        
        # TTS Engine - initialized on demand per thread or globally
        # We will init locally in thread to fix "only works once" issue
        
        # Model Initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.keypoint_classifier = KeyPointClassifier()
        
        # Read labels
        with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        # Setup UI
        self._setup_ui()
        
        # Camera Setup
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        # Start Processing
        self.process_frame()

    def _setup_ui(self):
        # Grid Configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main Container
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=0) # Title
        self.main_frame.grid_rowconfigure(1, weight=1) # Visuals
        self.main_frame.grid_rowconfigure(2, weight=0) # Text & Controls

        # 1. Header
        self.label_title = ctk.CTkLabel(self.main_frame, text="Sign Language To Text Conversion", font=("Roboto", 28, "bold"))
        self.label_title.grid(row=0, column=0, columnspan=3, pady=(10, 20))

        # 2. Visuals Area (Left: Camera, Center: Skeleton, Right: Chart)
        
        # Left: Camera Feed
        self.frame_camera = ctk.CTkFrame(self.main_frame)
        self.frame_camera.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.label_camera_title = ctk.CTkLabel(self.frame_camera, text="Live Feed", font=("Roboto", 16))
        self.label_camera_title.pack(pady=5)
        self.label_camera = ctk.CTkLabel(self.frame_camera, text="") # Image placeholder
        self.label_camera.pack(expand=True, fill="both", padx=5, pady=5)

        # Center: Skeleton Feed
        self.frame_skeleton = ctk.CTkFrame(self.main_frame)
        self.frame_skeleton.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.label_skeleton_title = ctk.CTkLabel(self.frame_skeleton, text="Hand Tracking", font=("Roboto", 16))
        self.label_skeleton_title.pack(pady=5)
        self.label_skeleton = ctk.CTkLabel(self.frame_skeleton, text="") 
        self.label_skeleton.pack(expand=True, fill="both", padx=5, pady=5)

        # Right: Reference Chart
        self.frame_chart = ctk.CTkFrame(self.main_frame)
        self.frame_chart.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        self.label_chart_title = ctk.CTkLabel(self.frame_chart, text="Reference Chart", font=("Roboto", 16))
        self.label_chart_title.pack(pady=5)
        
        # Load Chart Image
        try:
            chart_img = Image.open("assets/asl_chart.png")
            # Resize logic to fit
            chart_img.thumbnail((400, 400))
            self.chart_ctk_img = ctk.CTkImage(light_image=chart_img, dark_image=chart_img, size=chart_img.size)
            self.label_chart = ctk.CTkLabel(self.frame_chart, text="", image=self.chart_ctk_img)
            self.label_chart.pack(expand=True, pady=10)
        except Exception as e:
            self.label_chart = ctk.CTkLabel(self.frame_chart, text="Chart not found")
            self.label_chart.pack(expand=True)

        # 3. Controls & Text Area
        self.frame_controls = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frame_controls.grid(row=2, column=0, columnspan=3, padx=20, pady=20, sticky="ew")
        
        # Text Info
        self.frame_text = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_text.pack(side="left", fill="x", expand=True)

        self.label_char = ctk.CTkLabel(self.frame_text, text="Character: ", font=("Roboto", 20, "bold"))
        self.label_char.pack(anchor="w")
        
        self.label_sentence = ctk.CTkLabel(self.frame_text, text="Sentence: ", font=("Roboto", 20, "bold"))
        self.label_sentence.pack(anchor="w")

        # Suggestions
        self.frame_suggestions = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_suggestions.pack(side="left", padx=20)
        
        self.lbl_suggestions = ctk.CTkLabel(self.frame_suggestions, text="Suggestions:", font=("Roboto", 16, "bold"), text_color="#E74C3C")
        self.lbl_suggestions.pack(side="left", padx=5)

        # Helper to create buttons
        for i in range(4):
            btn = ctk.CTkButton(self.frame_suggestions, text="", width=80, command=lambda x=i: self.use_suggestion(x))
            btn.pack(side="left", padx=5)
            # Hide initially
            btn.pack_forget()
            self.suggestion_buttons.append(btn)
        
        self.update_suggestions() 

        # Action Buttons
        self.frame_actions = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_actions.pack(side="right")

        self.btn_clear = ctk.CTkButton(self.frame_actions, text="Clear", width=100, height=40, command=self.clear_text)
        self.btn_clear.pack(side="left", padx=10)

        self.btn_speak = ctk.CTkButton(self.frame_actions, text="Speak", width=100, height=40, command=self.speak_text)
        self.btn_speak.pack(side="left", padx=10)

    def process_frame(self):
        ret, image = self.cap.read()
        if not ret:
            self.after(10, self.process_frame)
            return

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        
        # Process Image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        skeleton_img = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255 # White background
        
        predicted_char = None
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculations
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Prediction
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                predicted_char = self.keypoint_classifier_labels[hand_sign_id]
                
                # Draw on skeleton image
                skeleton_img = draw_landmarks_on_white((image.shape[0], image.shape[1], 3), landmark_list)
        
        # --- Smoothing & Auto-Append Logic ---
        if predicted_char:
            self.prediction_history.append(predicted_char)
        else:
            self.prediction_history.append(None) # Clear or keep? Better to not append if no hand?
            # actually if no hand is detected, we should probably break the stability chain.
            self.prediction_history.clear()
            self.last_stable_char = ""
        
        smoothed_char = ""
        
        if len(self.prediction_history) == self.history_length:
            # Get most common element, ignoring Nones
            clean_hist = [c for c in self.prediction_history if c is not None]
            if clean_hist:
                most_common = Counter(clean_hist).most_common(1)
                # If the most common appears in > 60% of frames
                if most_common[0][1] > (self.history_length * 0.6):
                    smoothed_char = most_common[0][0]

        # Update Display with Smoothed Char
        if smoothed_char:
             self.label_char.configure(text=f"Character: {smoothed_char}")
             
             # Calculate Stability Duration
             if smoothed_char == self.last_stable_char:
                 duration = time.time() - self.stable_char_start_time
                 
                 # Visual feedback of timer? (Optional, maybe later)
                 # self.label_char.configure(text=f"Character: {smoothed_char} ({duration:.1f}s)")
                 
                 if duration > self.auto_add_threshold and not self.char_added_flag:
                     self.add_to_sentence(smoothed_char)
                     self.char_added_flag = True
             else:
                 self.last_stable_char = smoothed_char
                 self.stable_char_start_time = time.time()
                 self.char_added_flag = False
        else:
            self.label_char.configure(text="Character: ")
        
        # Update Camera Feed
        img_cam = cv.resize(debug_image, (400, 300))
        img_cam = cv.cvtColor(img_cam, cv.COLOR_BGR2RGB)
        img_cam_pil = Image.fromarray(img_cam)
        cam_ctk_img = ctk.CTkImage(light_image=img_cam_pil, dark_image=img_cam_pil, size=(400, 300))
        self.label_camera.configure(image=cam_ctk_img)
        self.label_camera.image = cam_ctk_img

        # Update Skeleton Feed
        img_skel = cv.resize(skeleton_img, (400, 300))
        img_skel = cv.cvtColor(img_skel, cv.COLOR_BGR2RGB) 
        img_skel_pil = Image.fromarray(img_skel)
        skel_ctk_img = ctk.CTkImage(light_image=img_skel_pil, dark_image=img_skel_pil, size=(400, 300))
        self.label_skeleton.configure(image=skel_ctk_img)
        self.label_skeleton.image = skel_ctk_img

        self.after(10, self.process_frame)

    def add_to_sentence(self, char):
         # If the last word was being typed, we might want to continue it?
         # For now, simplistic appending
         self.current_sentence += char
         self.label_sentence.configure(text=f"Sentence: {self.current_sentence}")
         self.update_suggestions()

    def clear_text(self):
        self.current_sentence = ""
        self.label_sentence.configure(text=f"Sentence: ")
        self.last_char = ""
        self.update_suggestions()

    def speak_text(self):
        text_to_speak = self.current_sentence
        if text_to_speak:
            def speak():
                try:
                    # Initialize engine newly for each thread to avoid loop locks
                    engine = pyttsx3.init()
                    engine.say(text_to_speak)
                    engine.runAndWait()
                    # engine.stop() # Usually not needed if runAndWait finishes
                except Exception as e:
                    print(f"TTS Error: {e}")
            
            threading.Thread(target=speak, daemon=True).start()
    
    def update_suggestions(self):
        # 1. Identify the last "word word" being formed.
        #    If sentence is "HELL", we assume the word is "HELL"
        #    If sentence is "HELLO W", we assume word is "W"
        
        words = self.current_sentence.split(" ")
        current_fragment = words[-1].upper() if words else ""
        
        matches = []
        if current_fragment:
            # Filter common words
            matches = [w for w in self.common_words if w.startswith(current_fragment)]
            # If exact match exists, maybe exclude it or show it (to complete/confirm?)
            # or show next likely words? For now prefix match is good.
        else:
            # Default suggestions
            matches = ["HELLO", "YES", "NO", "THANK"]
            
        display_suggestions = matches[:4]
        
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(display_suggestions):
                btn.configure(text=display_suggestions[i])
                btn.pack(side="left", padx=5)
            else:
                btn.pack_forget()

    def use_suggestion(self, index):
        word = self.suggestion_buttons[index].cget("text")
        
        # Replace the current fragment with the full word
        words = self.current_sentence.split(" ")
        if words:
             words[-1] = word # Replace last fragment
        else:
             words = [word]
             
        # Add a space after?
        self.current_sentence = " ".join(words) + " "
        
        self.label_sentence.configure(text=f"Sentence: {self.current_sentence}")
        self.update_suggestions()

if __name__ == "__main__":
    app = ASLApp()
    app.mainloop()
