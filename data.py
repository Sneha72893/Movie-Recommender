import os
import requests
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "ml-100k")
RATINGS_FILE = os.path.join(EXTRACTED_DIR, "u.data")
MOVIES_FILE = os.path.join(EXTRACTED_DIR, "u.item")

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading MovieLens 100k from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(ZIP_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
        
    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")

def load_data():
    download_and_extract()
    
    # Load ratings
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(RATINGS_FILE, sep='\t', names=ratings_cols, encoding='latin-1')
    
    # Load movies
    movies = pd.read_csv(MOVIES_FILE, sep='|', names=['item_id', 'title'], usecols=[0, 1], encoding='latin-1')
    
    return ratings, movies

def preprocess_data(ratings, movies):
    # Create mappings for user and item IDs to contiguous integers
    user_ids = ratings['user_id'].unique()
    item_ids = ratings['item_id'].unique()
    
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: idx for idx, i in enumerate(item_ids)}
    
    # For inference, map idx to original item ID and then to title
    idx2item = {idx: i for i, idx in item2idx.items()}
    item2title = dict(zip(movies['item_id'], movies['title']))
    
    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['item_idx'] = ratings['item_id'].map(item2idx)
    
    # Save mappings for the web app
    mappings = {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2item': idx2item,
        'item2title': item2title,
        'num_users': len(user_ids),
        'num_items': len(item_ids)
    }
    
    with open('mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
        
    return ratings, mappings

def get_train_test_split(ratings, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train_df, test_df

if __name__ == "__main__":
    ratings, movies = load_data()
    ratings, mappings = preprocess_data(ratings, movies)
    train_df, test_df = get_train_test_split(ratings)
    print(f"Data processed successfully. Num users: {mappings['num_users']}, Num items: {mappings['num_items']}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
