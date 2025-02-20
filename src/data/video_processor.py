import cv2
import os

class VideoProcessor:
    def __init__(self, input_dir, output_dir, target_size=224):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        os.makedirs(output_dir, exist_ok=True)
        
    def process_videos(self):
        """Process all videos in input directory"""
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi')):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, self.input_dir)
                    output_path = os.path.join(self.output_dir, relative_path, file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self.process_single_video(input_path, output_path)

    def process_single_video(self, input_path, output_path):
        """Center crop and resize single video"""
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate crop dimensions
        size = min(width, height)
        x = (width - size) // 2
        y = (height - size) // 2
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (self.target_size, self.target_size)
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Crop and resize
            frame = frame[y:y+size, x:x+size]
            frame = cv2.resize(frame, (self.target_size, self.target_size))
            writer.write(frame)
            
        cap.release()
        writer.release()
