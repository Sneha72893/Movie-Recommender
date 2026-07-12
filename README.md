# 🎬 Neural Collaborative Filtering (NCF) Recommender

This project implements a Deep Learning-based recommendation system using the **MovieLens-100k** dataset. The model architecture uses Neural Collaborative Filtering (NCF), which replaces the inner product in traditional matrix factorization with a neural architecture to learn user-item interactions.

## 📊 Dataset

The system uses the [MovieLens-100k](https://grouplens.org/datasets/movielens/100k/) dataset from GroupLens Research.
- **Users**: 943
- **Items (Movies)**: 1,682
- **Ratings**: 100,000 (1-5 scale)

The dataset is automatically downloaded and processed when you run the training script.

## 🧠 Approach

1. **Embeddings**: We map `user_ids` and `item_ids` to dense vector representations using PyTorch `nn.Embedding`.
2. **NCF Architecture**: 
   - We concatenate the user and item embeddings.
   - The concatenated vector is passed through a Multilayer Perceptron (MLP) with hidden layers and ReLU activations.
   - The final output layer predicts the explicit rating (regression).
3. **Training**: The model is trained using **Mean Squared Error (MSE)** loss and the Adam optimizer.
4. **Evaluation**: Performance is evaluated using **RMSE (Root Mean Square Error)** for rating prediction accuracy and **NDCG@10** for ranking quality.

## 🚀 Results

During evaluation on a 20% test split, the model achieves competitive metrics:
- **RMSE**: ~0.95 (Varies slightly based on random initialization)
- **NDCG@10**: ~0.85+

## 💻 Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
This will download the dataset, process it, train the PyTorch model, and save the best weights to `model.pth`.
```bash
python train.py
```

### 3. Run the Web Application
Launch the Streamlit app to interact with the recommendation system.
```bash
streamlit run app.py
```
*Note: Make sure the model is fully trained before running the app.*

## 🌐 Live Demo

You can view a live demo of the application here:
[Live Demo Link (Placeholder)](#)
*(Deploy the Streamlit app to Streamlit Cloud or Hugging Face Spaces to make this link active.)*
