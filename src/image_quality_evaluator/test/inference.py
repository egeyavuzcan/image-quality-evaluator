#!/usr/bin/env python
# coding: utf-8

"""
Script to perform inference using a trained image quality regression model.
Loads a model checkpoint and predicts quality scores for input images.
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from pathlib import Path

# --- Configuration ---

# Determine Project Root Directory dynamically (still needed for default model path)
# Assuming this script is in 'src/image_quality_evaluator/test/'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default path to the saved model (relative to project root)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "best_quality_model.pth"

# Model configuration (should match the trained model)
MODEL_NAME = "resnet50"
IMAGE_SIZE = 256
DROPOUT_RATE = 0.5 # Ensure this matches the dropout used during training

# --- Helper Functions ---

from ..utils.utils import preprocess_image # Import the shared function

def load_model(model_path, device):
    """Loads the trained ResNet50 model with the correct final layer."""
    print(f"Loading model from: {model_path}")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # Initialize the base model architecture
    model = models.resnet50(weights=None) # Don't load default weights here

    # Reconstruct the final layer structure used during training
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(num_ftrs, 1)
    )

    try:
        # Load the saved state dictionary
        state_dict = torch.load(model_path, map_location=device)

        # Handle potential state_dict key mismatches (e.g., DataParallel training)
        if list(state_dict.keys())[0].startswith('module.'):
             print("Removing 'module.' prefix from state_dict keys (likely trained with DataParallel).")
             state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print("Model state_dict loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch between the model architecture defined here and the one saved in the checkpoint.")
        print(f"Ensure DROPOUT_RATE ({DROPOUT_RATE}) matches the training script.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval() # Set model to evaluation mode
    return model

def predict_quality(model, image_tensor):
    """Predicts the quality score for a preprocessed image tensor."""
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(image_tensor)
    # Return the predicted score as a float
    # .item() extracts the scalar value from a single-element tensor
    return output.item()

# --- Main Execution --- 

def main(args):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model_path = Path(args.model_path)
    model = load_model(model_path, device)

    # Process each image path provided
    print("\nStarting predictions...")
    for image_path_str in args.image_paths:
        image_path = Path(image_path_str)
        print(f"-- Processing: {image_path.name} --")
        if not image_path.is_file():
             print(f"   Skipping (not a file): {image_path}")
             continue

        # Preprocess image
        image_tensor = preprocess_image(image_path, IMAGE_SIZE, device)

        # Predict if preprocessing was successful
        if image_tensor is not None:
            predicted_score = predict_quality(model, image_tensor)
            print(f"   Predicted Quality Score: {predicted_score:.4f}")
        else:
            print(f"   Skipping prediction due to preprocessing error.")
        print("---")

    print("\nInference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image quality using a trained CNN model.")
    
    parser.add_argument(
        "image_paths", 
        metavar="IMAGE_PATH", 
        type=str, 
        nargs='+', # Accept one or more image paths
        help="Path(s) to the input image file(s)."
    )
    parser.add_argument(
        "-m", "--model-path", 
        type=str, 
        default=str(DEFAULT_MODEL_PATH), # Convert Path object to string for default
        help=f"Path to the trained model .pth file (default: {DEFAULT_MODEL_PATH})"
    )

    args = parser.parse_args()

    main(args)
