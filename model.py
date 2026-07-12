import torch
import torch.nn as nn

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
        
        # Final output layer to predict explicit rating
        self.output_layer = nn.Linear(input_dim, 1)
        
    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        vector = torch.cat([user_embed, item_embed], dim=-1)
        
        # Pass through MLP
        mlp_output = self.mlp(vector)
        
        # Predict rating
        prediction = self.output_layer(mlp_output)
        
        return prediction.squeeze()
