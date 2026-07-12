import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.metrics import mean_squared_error, ndcg_score

from data import load_data, preprocess_data, get_train_test_split
from model import NCF

class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def calculate_ndcg(model, test_df, k=10):
    model.eval()
    
    # We will compute NDCG per user and average it
    user_groups = test_df.groupby('user_idx')
    ndcg_scores = []
    
    with torch.no_grad():
        for user_idx, group in user_groups:
            if len(group) < 2:
                continue
                
            items = torch.tensor(group['item_idx'].values, dtype=torch.long)
            users = torch.tensor([user_idx] * len(items), dtype=torch.long)
            true_ratings = group['rating'].values.reshape(1, -1)
            
            predictions = model(users, items).numpy().reshape(1, -1)
            
            score = ndcg_score(true_ratings, predictions, k=k)
            ndcg_scores.append(score)
            
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def train(epochs=10, batch_size=256, lr=0.001):
    print("Loading and preparing data...")
    ratings, movies = load_data()
    ratings, mappings = preprocess_data(ratings, movies)
    train_df, test_df = get_train_test_split(ratings)
    
    train_dataset = MovieLensDataset(train_df)
    test_dataset = MovieLensDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = NCF(num_users=mappings['num_users'], num_items=mappings['num_items']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_rmse = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for users, items, true_ratings in train_loader:
            users, items, true_ratings = users.to(device), items.to(device), true_ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, true_ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(users)
            
        train_loss /= len(train_loader.dataset)
        
        # Evaluation
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for users, items, true_ratings in test_loader:
                users, items = users.to(device), items.to(device)
                predictions = model(users, items)
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend(true_ratings.numpy())
                
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(test_targets, test_predictions))
        
        # Calculate NDCG@10
        model.to('cpu')
        ndcg_10 = calculate_ndcg(model, test_df, k=10)
        model.to(device)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - RMSE: {rmse:.4f} - NDCG@10: {ndcg_10:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            print("Saving best model...")
            torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
