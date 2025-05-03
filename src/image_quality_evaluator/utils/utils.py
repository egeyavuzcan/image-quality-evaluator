#!/usr/bin/env python
# coding: utf-8

"""
Utility functions and classes for the Image Quality Evaluator project.
"""

import random
import numpy as np
import torch
import re
import traceback
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torch.utils.data import Dataset

# --- Constants ---

# Image file extensions to consider
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

# Standard normalization values for ImageNet pre-trained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NORMALIZE_TRANSFORM = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# --- Utility Functions ---

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (can slightly slow down training)
        # Note: Setting deterministic can sometimes impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_score_from_folder(folder_name):
    """Extracts the midpoint score from folder names like 'score_X_to_Y'."""
    match = re.search(r'score_(\d+)_to_(\d+)', folder_name)
    if match:
        lower = int(match.group(1))
        upper = int(match.group(2))
        return (lower + upper) / 2.0
    return None # Return None if format doesn't match

def get_base_transforms(image_size):
    """Returns basic transformations (Resize, ToTensor, Normalize) used for validation/inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        NORMALIZE_TRANSFORM,
    ])

def get_train_transforms(image_size):
    """Returns transformations including data augmentation for training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), # Flip images horizontally randomly
        transforms.RandomRotation(15),     # Rotate images randomly by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adjust color randomly
        transforms.ToTensor(),             # Convert PIL Image to PyTorch Tensor
        NORMALIZE_TRANSFORM,               # Normalize tensor values
    ])

def preprocess_image(image_path, image_size, device):
    """Loads, preprocesses (using base transforms), and moves a single image to the specified device."""
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except UnidentifiedImageError:
        print(f"Error: Could not read image file (corrupt?): {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    preprocess = get_base_transforms(image_size)

    # Apply transformations and add batch dimension (unsqueeze(0))
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor

# --- Dataset Class ---

class ImageQualityDataset(Dataset):
    """Custom PyTorch Dataset for loading images and quality scores."""
    def __init__(self, image_paths, scores, transform=None):
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform
        if len(self.image_paths) != len(self.scores):
             raise ValueError("Number of image paths and scores must be the same.")
        #print(f"Dataset initialized with {len(self.image_paths)} samples.") # Moved print to main script

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self.image_paths):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.image_paths)}")

        img_path = self.image_paths[idx]
        score = torch.tensor(self.scores[idx], dtype=torch.float32)

        try:
            # Open image and ensure it's in RGB format
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Return image and score (unsqueezed to have shape [1])
            return image, score.unsqueeze(0)
        except FileNotFoundError:
            print(f"Warning: File not found: {img_path}. Skipping index {idx}.")
            return None # Return None for collate_fn to handle
        except UnidentifiedImageError:
             print(f"Warning: Could not read image (corrupt?): {img_path}. Skipping index {idx}.")
             return None # Return None for collate_fn to handle
        except Exception as e:
            print(f"Error loading image {img_path} at index {idx}: {e}")
            traceback.print_exc() # Print full traceback for unexpected errors
            return None # Return None for collate_fn to handle

# --- Custom Collate Function ---

def collate_fn_skip_none(batch):
    """Collate function that filters out None results from dataset's __getitem__."""
    # Filter out samples where __getitem__ returned None (due to errors)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the entire batch was problematic
    # Use default collate for the valid samples
    return torch.utils.data.dataloader.default_collate(batch)
