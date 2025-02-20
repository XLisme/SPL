import yaml
import os

class Config:
    def __init__(self, config_path='config.yml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get(self, key, default=None):
        """Get config value with optional default"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        return self.config[key]
