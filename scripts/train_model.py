from src.models.gan_trainer import GANTrainer
from torch.utils.data import DataLoader
import yaml
from src.data.pose_dataset import PoseDataset

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    train_dataset = PoseDataset('data/augmented_poses/train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Initialize trainer
    trainer = GANTrainer(config)
    
    # Train model
    trainer.train(train_loader, config['num_epochs'])

if __name__ == '__main__':
    main()
