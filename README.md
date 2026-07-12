# 🎬 AI Movie Recommender — Extreme Polyglot Microservices

A deep learning–based recommendation system built as a **real-world, industry-grade polyglot microservices architecture**. This project demonstrates the ability to use **four languages** and frameworks simultaneously, each chosen for what it does best.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               React + Vite  (JavaScript)                    │
│                   Frontend :5173                            │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Go API Gateway  (net/http)                     │
│                    :8080                                    │
│   Routes /api/recommendations → Python                     │
│   Routes /api/movies          → Rust                       │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐        ┌─────────────────────────┐
│  Python FastAPI     │        │  Rust Actix-Web         │
│  Inference Service  │        │  Data Engine            │
│       :8000         │        │       :3000             │
│  PyTorch NCF model  │        │  MovieLens in-memory    │
└─────────────────────┘        └─────────────────────────┘
```

| Service | Language | Framework | Port | Role |
|---|---|---|---|---|
| Frontend | JavaScript | React + Vite | 5173 | User Interface |
| API Gateway | **Go** | net/http | 8080 | Request Routing & BFF |
| Inference Service | **Python** | FastAPI | 8000 | NCF ML Predictions |
| Data Engine | **Rust** | Actix-Web | 3000 | Movie Metadata Store |

## 📊 Dataset
[MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) — 943 users, 1,682 movies, 100,000 ratings.

## 🧠 Approach: Neural Collaborative Filtering (NCF)
- User and Item **Embedding** layers → Concatenation → **MLP** → Rating prediction
- **RMSE**: `0.9728` | **NDCG@10**: `0.9107`

## 🚀 How to Run (4 terminals)

**1. Train the model first (one-time)**
```bash
pip install -r requirements.txt
python train.py
```

**2. Terminal 1 — Python Inference Service**
```bash
uvicorn api:app --reload --port 8000
```

**3. Terminal 2 — Rust Data Engine**
```bash
cd data-engine
cargo run --release
```

**4. Terminal 3 — Go API Gateway**
```bash
cd gateway
go run main.go
```

**5. Terminal 4 — React Frontend**
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## 📁 Project Structure
```
├── api.py              # 🐍 Python FastAPI — ML Inference Service
├── model.py            # 🐍 PyTorch NCF Model
├── train.py            # 🐍 Training Script
├── data.py             # 🐍 Data Preprocessing
├── requirements.txt    # 🐍 Python dependencies
├── gateway/
│   └── main.go         # 🐹 Go API Gateway
├── data-engine/
│   ├── src/main.rs     # 🦀 Rust Actix-Web Data Engine
│   └── Cargo.toml
└── frontend/
    └── src/
        ├── App.jsx     # ⚛️ React Component
        └── index.css   # ⚛️ Vanilla CSS Styles
```
