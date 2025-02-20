import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import os
from tqdm import tqdm
import yaml

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1):
    """Define the Generator architecture"""
    if netG == 'global':
        net = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global)
    elif netG == 'local':
        net = LocalEnhancer(input_nc, output_nc, ngf, n_local_enhancers, n_blocks_global)
    return net

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9):
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, padding=0), activation]
        
        # Downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                     nn.InstanceNorm2d(ngf * mult * 2),
                     activation]

        # Residual blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1),
                     nn.InstanceNorm2d(int(ngf * mult / 2)),
                     activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x) 

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_local_enhancers=1, n_blocks_local=3):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        # Global Generator
        ngf_global = ngf * (2**n_local_enhancers)
        self.global_gen = GlobalGenerator(input_nc, output_nc, ngf_global)

        # Local Enhancer
        for n in range(1, n_local_enhancers+1):
            ngf_local = ngf * (2**(n_local_enhancers-n))
            local_enhancer = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_local, kernel_size=7, padding=0),
                nn.InstanceNorm2d(ngf_local),
                nn.ReLU(True),
                nn.Conv2d(ngf_local, ngf_local*2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf_local*2),
                nn.ReLU(True),
                *[ResnetBlock(ngf_local*2) for _ in range(n_blocks_local)],
                nn.ConvTranspose2d(ngf_local*2, ngf_local, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf_local),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf_local, output_nc, kernel_size=7, padding=0),
                nn.Tanh()
            )
            setattr(self, f'local_enhancer_{n}', local_enhancer)

    def forward(self, input):
        output = self.global_gen(input)
        for n in range(1, self.n_local_enhancers+1):
            local_enhancer = getattr(self, f'local_enhancer_{n}')
            output = local_enhancer(output)
        return output

class PoseExtractor:
    def __init__(self, output_dir="temp_poses"):
        # Khởi tạo MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Khởi tạo models với các tham số phù hợp
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
        dirs = ['poses', 'faces', 'keypoints']
        for d in dirs:
            path = os.path.join(self.output_dir, d)
            if not os.path.exists(path):
                os.makedirs(path)

    def extract_frames(self, video_path, num_frames=30):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames

    def create_stick_figure(self, landmarks, image_shape):
        """Create stick figure from pose landmarks"""
        h, w = image_shape[:2]
        stick_figure = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            stick_figure,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,0), thickness=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,0,0), thickness=2)
        )
        
        return stick_figure

    def get_face_box(self, landmarks, image_shape):
        """Extract face bounding box from landmarks"""
        h, w = image_shape[:2]
        face_landmarks = [landmarks.landmark[i] for i in [0,1,2,3,4,5,6,7,8,9,10]]
        
        x_coords = [int(lm.x * w) for lm in face_landmarks]
        y_coords = [int(lm.y * h) for lm in face_landmarks]
        
        minx = max(0, min(x_coords) - 20)
        maxx = min(w, max(x_coords) + 20)
        miny = max(0, min(y_coords) - 20)
        maxy = min(h, max(y_coords) + 20)
        
        # Make square box
        box_size = 128
        center_x = (minx + maxx) // 2
        center_y = (miny + maxy) // 2
        
        minx = max(0, center_x - box_size//2)
        maxx = min(w, center_x + box_size//2)
        miny = max(0, center_y - box_size//2)
        maxy = min(h, center_y + box_size//2)
        
        return [miny, maxy, minx, maxx]

    def process_frame(self, frame):
        """Process single frame to extract pose and hand data"""
        # Chuyển đổi ảnh từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý hands
        hand_results = self.hands.process(rgb_frame)
        
        # Xử lý pose
        pose_results = self.pose.process(rgb_frame)
        
        current_landmarks = []
        
        # Lấy hand landmarks nếu có
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    current_landmarks.append((landmark.x, landmark.y, landmark.z))

        # Lấy pose landmarks nếu có
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks
            # Lấy các điểm upper body từ 11 đến 14
            upper_body_landmarks = [
                pose_landmarks.landmark[11],  # Left shoulder
                pose_landmarks.landmark[12],  # Right shoulder
                pose_landmarks.landmark[13],  # Left elbow
                pose_landmarks.landmark[14],  # Right elbow
            ]
            for landmark in upper_body_landmarks:
                current_landmarks.append((landmark.x, landmark.y, landmark.z))

        return current_landmarks, pose_results, hand_results

    def euclidean_distance_3d(self, p1, p2):
        """Calculate 3D Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

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
                
            current_landmarks, pose_results, hand_results = self.process_frame(frame)
            
            # Tính total Euclidean distance
            total_distance = 0
            if prev_landmarks is not None and len(current_landmarks) > 0:
                for p1, p2 in zip(prev_landmarks, current_landmarks):
                    total_distance += self.euclidean_distance_3d(p1, p2)
            
            # Tạo stick figure và lưu kết quả
            if pose_results.pose_landmarks:
                stick_figure = self.create_stick_figure(pose_results.pose_landmarks, frame.shape)
                face_box = self.get_face_box(pose_results.pose_landmarks, frame.shape)
                pose_data.append((stick_figure, face_box, current_landmarks))
            
            prev_landmarks = current_landmarks
            
        cap.release()
        return pose_data

class PoseGANProcessor:
    def __init__(self, opt): 
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_networks()
        
    def initialize_networks(self):
        """Initialize GAN networks"""
        # Generator
        self.netG = define_G(
            input_nc=self.opt['input_nc'],
            output_nc=self.opt['output_nc'],
            ngf=self.opt['ngf'],
            netG=self.opt['netG'],
            n_downsample_global=self.opt['n_downsample_global'],
            n_blocks_global=self.opt['n_blocks_global'],
            n_local_enhancers=self.opt['n_local_enhancers']
        ).to(self.device)
        
        # Face Generator if enabled
        if self.opt['face_generator']:
            self.faceGen = define_G(
                input_nc=3,
                output_nc=3,
                ngf=32,
                netG='local'
            ).to(self.device)
            
        # Load pre-trained weights
        self.load_networks()

    def load_networks(self):
        """Load pre-trained weights"""
        self.netG.load_state_dict(torch.load(self.opt['generator_path']))
        if self.opt['face_generator']:
            self.faceGen.load_state_dict(torch.load(self.opt['face_generator_path']))

    def process_pose_sequence(self, pose_data):
        """Process sequence of poses to generate video frames"""
        results = []
        prev_frame = None
        
        for i in range(len(pose_data)-1):
            curr_pose, curr_face, curr_keypoints = pose_data[i]
            next_pose, next_face, next_keypoints = pose_data[i+1]
            
            # Convert to torch tensors
            curr_pose = torch.from_numpy(curr_pose).float().to(self.device)
            curr_pose = curr_pose.permute(2, 0, 1).unsqueeze(0)
            
            # Generate frame
            with torch.no_grad():
                if prev_frame is None:
                    prev_frame = torch.zeros_like(curr_pose)
                
                # Generate current frame
                input_concat = torch.cat((curr_pose, prev_frame), dim=1)
                generated_frame = self.netG(input_concat)
                
                # Enhance face if enabled
                if self.opt['face_generator']:
                    generated_frame = self.enhance_face_details(generated_frame, curr_face)
                
                # Apply temporal smoothing
                if i > 0:
                    generated_frame = 0.8 * generated_frame + 0.2 * prev_frame
                
                prev_frame = generated_frame
                results.append(self.postprocess_frame(generated_frame))
        
        return results

    def enhance_face_details(self, frame, face_coords):
        """Enhance facial details"""
        miny, maxy, minx, maxx = face_coords
        face_region = frame[:, :, miny:maxy, minx:maxx]
        
        # Resize to standard size
        face_region = F.interpolate(
            face_region,
            size=(self.opt['face_dim'], self.opt['face_dim']),
            mode='bilinear',
            align_corners=True
        )
        
        # Generate enhanced face
        enhanced_face = self.faceGen(face_region)
        
        # Resize back and blend
        enhanced_face = F.interpolate(
            enhanced_face,
            size=(maxy-miny, maxx-minx),
            mode='bilinear',
            align_corners=True
        )
        
        frame[:, :, miny:maxy, minx:maxx] = enhanced_face
        return frame

    def postprocess_frame(self, frame):
        """Post-process generated frame"""
        # Convert to numpy and adjust range
        frame = frame.cpu().numpy()[0].transpose(1, 2, 0)
        frame = (frame * 255).astype(np.uint8)
        
        # Color correction
        frame = cv2.convertScale(frame, alpha=1.1, beta=0)  # Brightness
        frame = cv2.convertScale(frame, alpha=1.2, beta=0)  # Contrast
        
        return frame

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pose extractor
    pose_extractor = PoseExtractor(output_dir="temp_poses")
    
    # Process video to get pose data
    video_path = "input_video.mp4"
    pose_data = pose_extractor.process_video(video_path)
    
    # Kiểm tra highest_y để lọc các frame không cần thiết
    filtered_pose_data = []
    mfs_count = 0
    mfs_frame_count = 0
    
    for data in pose_data:
        _, _, landmarks = data
        if landmarks:
            highest_y = min(lm[1] for lm in landmarks if isinstance(lm, tuple))
            if highest_y <= 0.75:  # Chỉ giữ lại các frame có highest_y <= 0.75
                filtered_pose_data.append(data)
            else:
                mfs_frame_count += 1
                if mfs_frame_count >= 10:
                    mfs_count += 1
                    if mfs_count >= 2:
                        break
        
    # Initialize GAN processor và tiếp tục xử lý
    gan_processor = PoseGANProcessor(config)
    generated_frames = gan_processor.process_pose_sequence(filtered_pose_data)
    
    # Save output video
    output_path = "output_video.mp4"
    fps = 30
    h, w = generated_frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for frame in generated_frames:
        writer.write(frame)
    writer.release()
    
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()