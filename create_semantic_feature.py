import os
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, ShuffleSplit
import numpy as np
import csv
import torch
import clip
from PIL import Image
from sklearn.metrics import pairwise_distances
from tqdm import tqdm  # Import tqdm for progress tracking

def prepare_clip_features(batch_size=16):
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the CLIP model
    model, preprocess = clip.load("ViT-L/14", device=device)

    # Read the CSV file
    csv_filepath = "/home/dank/sethust/fil/AIGC/AIGI/AI-1k/AIGC_MOS_Zscore_full.csv" #for prompts
    image_dir = "/home/dank/sethust/fil/AIGC/AIGI/AI-1k/images" #for images
    data_dir = 'test_AI1k'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    image_filenames = []
    prompts = []

    with open(csv_filepath, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)  # Skip header row
        for row in csvreader:
            image_filenames.append(os.path.join(image_dir, row[0]))
            prompts.append(row[1])

    # Preprocess images and tokenize prompts
    num_images = len(image_filenames)
    num_batches = (num_images + batch_size - 1) // batch_size  # Calculate the number of batches
    all_image_features = []
    all_text_features = []

    print(f"num_batches: {num_batches}")
    for i in tqdm(range(num_batches), desc="Processing images", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_images)

        # Load and preprocess images for the current batch
        batch_images = torch.stack([preprocess(Image.open(image_filename)).to(device) 
                                    for image_filename in image_filenames[start_idx:end_idx]])

        # Tokenize prompts for the current batch
        batch_prompts = prompts[start_idx:end_idx]
        batch_texts = clip.tokenize(batch_prompts).to(device)

        # Compute image and text features for the current batch
        with torch.no_grad():
            batch_image_features = model.encode_image(batch_images)
            batch_text_features = model.encode_text(batch_texts)

        all_image_features.append(batch_image_features)
        all_text_features.append(batch_text_features)

    # Concatenate features from all batches
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)

    # Save features to files
    np.save(os.path.join(data_dir, 'all_clip_image_features.npy'), image_features.cpu().numpy())
    np.save(os.path.join(data_dir, 'all_clip_text_features.npy'), text_features.cpu().numpy())

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (image_features + text_features).cpu().numpy()
    np.save(os.path.join(data_dir, 'all_similarity.npy'), similarity)

prepare_clip_features(batch_size=16)  
