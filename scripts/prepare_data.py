from src.data.video_processor import VideoProcessor
from src.data.pose_extractor import PoseExtractor
from src.data.data_augmentor import DataAugmentor
import yaml

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Process videos
    video_processor = VideoProcessor(
        input_dir='data/raw_videos',
        output_dir='data/processed_videos'
    )
    video_processor.process_videos()
    
    # Extract poses
    pose_extractor = PoseExtractor(output_dir='data/poses')
    pose_extractor.process_directory('data/processed_videos')
    
    # Augment data
    augmentor = DataAugmentor(
        input_dir='data/poses',
        output_dir='data/augmented_poses'
    )
    augmentor.augment_all()

if __name__ == '__main__':
    main()
