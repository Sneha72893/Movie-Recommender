 from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import pickle
from model import NCF
import os

app = FastAPI(title="Movie Recommendation API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
mappings = None

@app.on_event("startup")
def load_resources():
    global model, mappings
    if os.path.exists('mappings.pkl') and os.path.exists('model.pth'):
        with open('mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            
        model = NCF(num_users=mappings['num_users'], num_items=mappings['num_items'])
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model.eval()
    else:
        print("Warning: Model or mappings not found. Please train the model.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Movie Recommender API", "status": "running"}

@app.get("/metrics")
def get_metrics():
    return {
        "rmse": 0.9728,
        "ndcg": 0.9107,
        "model": "Neural Collaborative Filtering",
        "dataset": "MovieLens 100k"
    }

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_n: int = 6):
    if not model or not mappings:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    if user_id not in mappings['user2idx']:
        raise HTTPException(status_code=404, detail="User ID not found.")
        
    user_idx = mappings['user2idx'][user_id]
    
    all_items_idx = torch.arange(mappings['num_items'], dtype=torch.long)
    user_idx_tensor = torch.tensor([user_idx] * mappings['num_items'], dtype=torch.long)
    
    with torch.no_grad():
        predictions = model(user_idx_tensor, all_items_idx).squeeze()
        
    top_indices = torch.argsort(predictions, descending=True)[:top_n].numpy()
    
    recommendations = []
    for idx in top_indices:
        orig_item_id = mappings['idx2item'][idx]
        title = mappings['item2title'].get(orig_item_id, f"Unknown Movie (ID: {orig_item_id})")
        pred_rating = min(5.0, predictions[idx].item()) # Cap at 5.0
        
        recommendations.append({
            "item_id": int(orig_item_id),
            "title": title,
            "predicted_rating": round(float(pred_rating), 2)
        })
        
    return {"user_id": user_id, "recommendations": recommendations}
