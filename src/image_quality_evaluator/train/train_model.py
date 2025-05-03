#!/usr/bin/env python
# coding: utf-8

"""
Script to train a CNN model (ResNet50) for image quality regression.
This version includes data augmentation, dropout, weight decay, 
and learning rate scheduling.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm
import time
import re
import traceback

# --- Configuration --- 

# Determine Project Root Directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Relative Paths (assuming 'data' and 'output' folders in project root)
DATASET_DIR = PROJECT_ROOT / "data" / "Image_Dataset_Manual" # <--- Adjust subfolder if needed
OUTPUT_DIR = PROJECT_ROOT / "output"
BEST_MODEL_PATH = OUTPUT_DIR / "best_quality_model.pth"

# Model & Training Hyperparameters
MODEL_NAME = "resnet50"
IMAGE_SIZE = 256       # Resize images to this size (squared)
BATCH_SIZE = 16        # Adjust based on GPU memory
NUM_EPOCHS = 50        # Max epochs; early stopping might terminate sooner
LEARNING_RATE = 1e-4   # Initial learning rate
VALIDATION_SPLIT = 0.2 # Percentage of data for validation
RANDOM_SEED = 42       # For reproducibility
NUM_WORKERS = 0        # Set to 0 for Windows/debugging, >0 for Linux/performance
EARLY_STOPPING_PATIENCE = 7 # Epochs to wait for improvement before stopping
DROPOUT_RATE = 0.5     # Dropout probability before the final layer
WEIGHT_DECAY = 1e-3    # L2 regularization strength for AdamW

# LR Scheduler Config
SCHEDULER_FACTOR = 0.1 # Factor to reduce LR by (new_lr = lr * factor)
SCHEDULER_PATIENCE = 3 # Epochs with no val_loss improvement before reducing LR
SCHEDULER_MIN_LR = 1e-6  # Minimum learning rate

from ..utils.utils import (
    set_seed, parse_score_from_folder, ImageQualityDataset, collate_fn_skip_none,
    get_train_transforms, get_base_transforms, SUPPORTED_EXTENSIONS
)

def load_data(dataset_dir):
    """Loads image paths and their corresponding scores from the dataset directory."""
    print(f"\nStep 1: Loading data...")
    print(f"Scanning dataset directory: {dataset_dir}...")
    image_paths = []
    scores = []
    folder_counts = {}

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Error: Dataset directory not found or is not a directory at {dataset_dir}")
        print("Please ensure the 'data/Image_Dataset_Manual' folder exists relative to the project root.")
        sys.exit(1) # Exit if data directory is missing

    for category_folder in dataset_dir.iterdir():
        if category_folder.is_dir():
            target_score = parse_score_from_folder(category_folder.name)
            if target_score is not None:
                print(f"  Processing folder: {category_folder.name} (Target Score: {target_score})")
                count = 0
                # Recursively search for image files
                for file_path in category_folder.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        image_paths.append(file_path)
                        scores.append(target_score)
                        count += 1
                folder_counts[category_folder.name] = count
                print(f"    Found {count} images in this folder (and subfolders).")
            else:
                print(f"  Skipping folder (name does not match 'score_X_to_Y' format): {category_folder.name}")
        else:
            print(f"  Skipping non-directory item: {category_folder.name}")

    total_images = sum(folder_counts.values())
    print(f"Finished scanning. Found {total_images} total supported images.")
    if not image_paths:
        print("Error: No images found in the dataset directory. Check the structure and file extensions.")
        sys.exit(1) # Exit if no images found

    return image_paths, scores

# --- Training & Validation Loops ---

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Performs one training epoch."""
    model.train() # Set model to training mode
    running_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False, unit="batch")
    for i, batch in enumerate(progress_bar):
        # Skip batch if collate_fn returned None
        if batch is None:
            print(f"Warning: Skipping an entire batch due to previous loading errors.")
            continue
        try:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            current_batch_size = inputs.size(0)
            total_samples += current_batch_size

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accumulate loss (weighted by batch size)
            running_loss += loss.item() * current_batch_size
            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        except Exception as e:
            print(f"\nError during training batch {i}: {e}")
            traceback.print_exc()
            # Optionally decide whether to continue or stop training on batch errors
            # raise e # Uncomment to stop training on the first batch error

    if total_samples == 0:
        print("Warning: No samples were successfully processed in this training epoch.")
        return 0.0

    epoch_loss = running_loss / total_samples
    return epoch_loss

def validate_one_epoch(model, dataloader, criterion, device):
    """Performs one validation epoch."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    running_mae = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False, unit="batch")
    with torch.no_grad(): # Disable gradient calculations for validation
        for i, batch in enumerate(progress_bar):
            # Skip batch if collate_fn returned None
            if batch is None:
                print(f"Warning: Skipping an entire validation batch due to previous loading errors.")
                continue
            try:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                current_batch_size = inputs.size(0)
                total_samples += current_batch_size

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean() # Mean Absolute Error

                # Accumulate metrics (weighted by batch size)
                running_loss += loss.item() * current_batch_size
                running_mae += mae.item() * current_batch_size
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae.item():.4f}")
            except Exception as e:
                 print(f"\nError during validation batch {i}: {e}")
                 traceback.print_exc()
                 # Optionally decide whether to continue or stop validation
                 # raise e # Uncomment to stop validation on the first batch error

    if total_samples == 0:
        print("Warning: No samples were successfully processed in this validation epoch.")
        return 0.0, 0.0

    epoch_loss = running_loss / total_samples
    epoch_mae = running_mae / total_samples
    return epoch_loss, epoch_mae


# --- Main Function ---

def main():
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Load Data
    image_paths, scores = load_data(DATASET_DIR)

    # 2. Split Data
    print("\nStep 2: Splitting data...")
    # Stratify split if possible to maintain score distribution
    try:
        train_paths, val_paths, train_scores, val_scores = train_test_split(
            image_paths, scores, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=scores
        )
    except ValueError: # Handle cases where stratification isn't possible
        print("Warning: Could not stratify data split (e.g., too few samples per class). Performing random split.")
        train_paths, val_paths, train_scores, val_scores = train_test_split(
            image_paths, scores, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
        )
    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")

    # 3. Define Transformations
    print("\nStep 3: Defining transformations...")
    # Use functions from utils
    train_transforms = get_train_transforms(IMAGE_SIZE)
    val_transforms = get_base_transforms(IMAGE_SIZE)
    print("Transformations defined.")

    # 4. Create Datasets and DataLoaders
    print("\nStep 4: Creating Datasets and DataLoaders...")
    train_dataset = ImageQualityDataset(train_paths, train_scores, transform=train_transforms)
    val_dataset = ImageQualityDataset(val_paths, val_scores, transform=val_transforms)
    print(f"Train dataset initialized with {len(train_dataset)} samples.") 
    print(f"Validation dataset initialized with {len(val_dataset)} samples.") 

    # Use the custom collate_fn to handle potential loading errors
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_none)
    print("Datasets and DataLoaders created.")

    # 5. Load Model
    print("\nStep 5: Loading pre-trained model (ResNet50) and modifying head...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace the final fully connected layer for regression
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE), # Add dropout for regularization
        nn.Linear(num_ftrs, 1)      # Output a single score
    )

    model = model.to(device)
    print("Model loaded and modified for regression.")

    # 6. Define Loss, Optimizer, and Scheduler
    print("\nStep 6: Defining loss function (MSE), optimizer (AdamW), and LR scheduler...")
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',              # Reduce LR when validation loss stops decreasing
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
        verbose=False          # Set to True to print message on LR reduction (check PyTorch version compatibility)
    )
    print("Loss, optimizer, and LR scheduler defined.")

    # 7. Training Loop
    print("\nStep 7: Starting training loop...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        val_loss, val_mae = validate_one_epoch(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss (MSE): {val_loss:.4f}")
        print(f"  Val MAE: {val_mae:.4f}")
        print(f"  Duration: {epoch_duration:.2f} seconds")

        # Step the LR scheduler based on validation loss
        scheduler.step(val_loss)

        # Checkpoint saving and Early Stopping
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Model saved to {BEST_MODEL_PATH}")
                epochs_no_improve = 0 # Reset counter
            except Exception as e:
                 print(f"Error saving model: {e}")
                 traceback.print_exc()
        else:
            print(f"Validation loss did not improve from {best_val_loss:.4f}")
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break # Exit training loop

    total_training_time = time.time() - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total training time: {total_training_time / 60:.2f} minutes")
    print(f"Best validation loss (MSE) achieved: {best_val_loss:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


# --- Script Execution ---

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"Exiting script: {e}") # Handle sys.exit calls gracefully
    except Exception as e:
        print(f"\nAn unexpected error occurred during script execution:")
        traceback.print_exc() # Print detailed traceback
