import numpy as np
import os

class DataAugmentor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def augment_all(self):
        """Augment all pose data in input directory"""
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.npy'):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, self.input_dir)
                    output_dir = os.path.join(self.output_dir, relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    self.augment_pose_sequence(input_path, output_dir)

    def augment_pose_sequence(self, input_path, output_dir):
        """Apply augmentation to single pose sequence"""
        pose_data = np.load(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Original data
        np.save(os.path.join(output_dir, f"{base_name}_orig.npy"), pose_data)
        
        # Random noise
        noised = self.add_noise(pose_data)
        np.save(os.path.join(output_dir, f"{base_name}_noise.npy"), noised)
        
        # Scale
        scaled = self.scale(pose_data)
        np.save(os.path.join(output_dir, f"{base_name}_scale.npy"), scaled)
        
        # Rotate
        rotated = self.rotate(pose_data)
        np.save(os.path.join(output_dir, f"{base_name}_rotate.npy"), rotated)

    def add_noise(self, pose_data, noise_factor=0.005):
        """Add random noise to pose data"""
        noise = np.random.normal(0, noise_factor, pose_data.shape)
        return pose_data + noise

    def scale(self, pose_data, scale_range=(0.9, 1.1)):
        """Scale pose data randomly"""
        scale = np.random.uniform(*scale_range)
        return pose_data * scale

    def rotate(self, pose_data, max_angle=15):
        """Rotate pose data randomly"""
        angle = np.random.uniform(-max_angle, max_angle)
        rad = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        
        # Rotate x,y coordinates (z remains unchanged)
        rotated = pose_data.copy()
        rotated[..., :2] = np.dot(pose_data[..., :2], rot_matrix.T)
        return rotated
