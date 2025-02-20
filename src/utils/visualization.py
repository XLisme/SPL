import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_comparison(self, original_frame, generated_frame, pose, index):
        """Save comparison of original, pose and generated frames"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(pose)
        axes[1].set_title('Pose')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(generated_frame, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Generated')
        axes[2].axis('off')
        
        plt.savefig(self.output_dir / f'comparison_{index}.png')
        plt.close()
        
    def create_video(self, frames, output_path, fps=30):
        """Create video from frames"""
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(self.output_dir / output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        for frame in frames:
            writer.write(frame)
        writer.release()
