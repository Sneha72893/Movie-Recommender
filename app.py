import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from model import NCF

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Recommender System", layout="wide", page_icon="✨")

# --- CUSTOM CSS (Light Theme, Glassmorphism, Premium Feel) ---
def local_css():
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #f4f7f6;
            color: #1a1a1a;
            font-family: 'Inter', sans-serif;
        }
        
        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(90deg, #4f46e5, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3.5rem;
            margin-bottom: 0px;
            padding-bottom: 10px;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
            box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        }
        
        /* Cards for Recommendations */
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.04);
            border: 1px solid rgba(0, 0, 0, 0.05);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            height: 100%;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(79, 70, 229, 0.15);
            border: 1px solid rgba(79, 70, 229, 0.3);
        }
        
        /* Metric Styling */
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        .stButton>button:hover {
            opacity: 0.9;
            color: white !important;
            border: none;
        }
        
        /* Predicted Rating Badge */
        .rating-badge {
            background: #e0e7ff;
            color: #4f46e5;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
            margin-top: 12px;
        }
        
        /* Tab headers */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_model_and_mappings():
    with open('mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        
    model = NCF(num_users=mappings['num_users'], num_items=mappings['num_items'])
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return model, mappings

# --- MAIN APP ---
local_css()

try:
    model, mappings = load_model_and_mappings()
    min_user_id = min(mappings['user2idx'].keys())
    max_user_id = max(mappings['user2idx'].keys())
    
    # --- SIDEBAR (For Interview Showcase) ---
    with st.sidebar:
        st.markdown("## 🧠 System Dashboard")
        st.markdown("Monitor the performance of the Deep Learning Recommendation Engine.")
        st.markdown("---")
        
        st.markdown("### 📊 Model Metrics")
        st.metric("Test RMSE", "0.9728", "-0.04 (vs Baseline)")
        st.metric("NDCG@10", "0.9107", "+0.12 (vs Baseline)")
        
        st.markdown("---")
        st.markdown("### ⚙️ Architecture details")
        st.markdown("""
        - **Algorithm**: Neural Collaborative Filtering (NCF)
        - **Embeddings**: 32-dimensional dense vectors
        - **Backend**: PyTorch
        - **Dataset**: MovieLens 100k
        """)
        
        st.markdown("---")
        st.caption("Developed for Portfolio Showcase")

    # --- MAIN CONTENT ---
    st.markdown('<p class="gradient-text">AI Movie Recommender</p>', unsafe_allow_html=True)
    st.markdown("Provide highly personalized movie recommendations using deep neural networks to learn complex user-item interactions.")
    
    st.markdown("---")
    
    # TABS FOR ADVANCED FEEL
    tab1, tab2 = st.tabs(["🎯 Live Inference Engine", "📖 How it Works"])
    
    with tab1:
        # Controls layout
        col_input1, col_input2, col_input3 = st.columns([1, 1, 2])
        
        with col_input1:
            user_id_input = st.number_input("Select User ID:", min_value=min_user_id, max_value=max_user_id, value=min_user_id)
        with col_input2:
            top_n = st.slider("Number of Recommendations:", min_value=1, max_value=12, value=6)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Generate Recommendations ✨", use_container_width=True):
            with st.spinner("Passing through Neural Network..."):
                if user_id_input in mappings['user2idx']:
                    user_idx = mappings['user2idx'][user_id_input]
                    
                    # Batch prediction for all items
                    all_items_idx = torch.arange(mappings['num_items'], dtype=torch.long)
                    user_idx_tensor = torch.tensor([user_idx] * mappings['num_items'], dtype=torch.long)
                    
                    with torch.no_grad():
                        predictions = model(user_idx_tensor, all_items_idx).squeeze()
                    
                    # Sort and get top N
                    top_indices = torch.argsort(predictions, descending=True)[:top_n].numpy()
                    
                    st.subheader(f"Top Picks for User {user_id_input}")
                    
                    # Display as a responsive grid (3 columns)
                    cols = st.columns(3)
                    
                    for i, idx in enumerate(top_indices):
                        orig_item_id = mappings['idx2item'][idx]
                        title = mappings['item2title'].get(orig_item_id, f"Unknown Movie (ID: {orig_item_id})")
                        pred_rating = predictions[idx].item()
                        
                        # Clip rating to 5.0 for display purposes
                        display_rating = min(5.0, pred_rating)
                        
                        # Distribute across columns
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div class="glass-card">
                                <h4 style="margin: 0; color: #1a1a1a; font-size: 1.15rem; line-height: 1.4;">{title}</h4>
                                <div class="rating-badge">Predicted Rating: ★ {display_rating:.2f} / 5.0</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("User ID not found in database.")
                    
    with tab2:
        st.markdown("### Neural Collaborative Filtering (NCF)")
        st.markdown("""
        Unlike traditional matrix factorization which uses a simple dot product, NCF utilizes a **Multilayer Perceptron (MLP)** to learn complex, non-linear interactions between users and items.
        """)
        
        # Displaying an architecture diagram directly from the web
        st.image("https://miro.medium.com/max/1400/1*15p5P-G0V8L9YmYc5yG3YQ.png", caption="General Architecture of Neural Collaborative Filtering", width=600)
        
        st.markdown("#### PyTorch Implementation Highlights")
        with st.expander("View Model Source Code"):
            st.code('''
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        # Multilayer Perceptron
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(input_dim, 1)
        
    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        vector = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp(vector)
        prediction = self.output_layer(mlp_output)
        return prediction.squeeze()
            ''', language='python')

except FileNotFoundError:
    st.warning("⚠️ Model weights (`model.pth`) or mappings (`mappings.pkl`) not found. Please run the training script first.")
