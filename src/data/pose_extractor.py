import cv2
import numpy as np
import mediapipe as mp
import os

class PoseExtractor:
    def __init__(self, output_dir="data/poses"):
        # Khởi tạo MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Khởi tạo models
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.output_dir = output_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.output_dir, exist_ok=True)

    def process_directory(self, input_dir):
        """Process all videos in a directory"""
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_dir)
                    output_path = os.path.join(self.output_dir, relative_path, 
                                             os.path.splitext(file)[0] + '.npy')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    pose_data = self.process_video(video_path)
                    if pose_data:
                        np.save(output_path, pose_data)

    def process_video(self, video_path, num_frames=30):
        """Process video and extract pose sequence data"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        pose_data = []
        prev_landmarks = None
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            current_landmarks, pose_results = self.process_frame(frame)
            
            if pose_results.pose_landmarks:
                pose_data.append(current_landmarks)
            
            prev_landmarks = current_landmarks
            
        cap.release()
        return np.array(pose_data) if pose_data else None

    def process_frame(self, frame):
        """Process single frame to extract pose and hand data"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        landmarks = []
        
        # Extract hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

        # Extract pose landmarks
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
                
        return np.array(landmarks), pose_results
