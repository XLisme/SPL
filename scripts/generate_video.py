import torch
from src.models.generator import Generator
from src.data.pose_extractor import PoseExtractor
from src.utils.visualization import Visualizer
from src.utils.config import Config
import cv2

def main():
    # Load config
    config = Config()
    
    # Initialize components
    generator = Generator(config)
    generator.load_state_dict(torch.load(config['generator_path']))
    generator.eval()
    
    pose_extractor = PoseExtractor()
    visualizer = Visualizer()
    
    # Process input video
    video_path = "data/raw_videos/input.mp4"
    pose_data = pose_extractor.process_video(video_path)
    
    # Generate frames
    generated_frames = []
    with torch.no_grad():
        for pose in pose_data:
            pose_tensor = torch.from_numpy(pose).unsqueeze(0)
            generated_frame = generator(pose_tensor)
            generated_frame = generated_frame.squeeze().numpy()
            generated_frames.append(generated_frame)
    
    # Create output video
    visualizer.create_video(generated_frames, 'output.mp4')

if __name__ == "__main__":
    main()
