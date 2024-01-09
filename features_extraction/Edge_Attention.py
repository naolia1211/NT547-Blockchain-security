import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import os
import numpy as np



# Dữ liệu giả định các cạnh
edge_index_data = [[0, 1], [1, 2], [2, 3]]  # Thay thế bằng cặp index thực tế của bạn

# Chuyển đổi danh sách các cạnh thành tensor edge_index
edge_indices = torch.tensor(edge_index_data, dtype=torch.long).t().contiguous()

# Số chiều của edge features
d_model = 100

# Số lượng heads trong Multi-head Attention
num_heads = 4

# Hàm Positional Encoding
def positional_encoding(length, d_model):
    position = torch.arange(length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Định nghĩa Edge Attention Module
class EdgeAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EdgeAttentionModule, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, edge_features):
        # Tính toán positional encoding
        pe = positional_encoding(edge_features.size(0), d_model).to(edge_features.device)
        # Mở rộng pe để có cùng số chiều batch với edge_features
        pe = pe.unsqueeze(0).repeat(edge_features.size(1), 1, 1)
        edge_features = edge_features + pe

        # Thêm một chiều batch và permute để chuẩn bị cho Multi-head Attention
        edge_features = edge_features.permute(1, 0, 2)

        # Multi-head Attention
        edge_features, _ = self.mha(edge_features, edge_features, edge_features)

        # Loại bỏ chiều batch và permute trở lại
        edge_features = edge_features.permute(1, 0, 2)

        # Feedforward MLP
        edge_features = self.mlp(edge_features)

        # Layer Norm cuối cùng
        semantic_features = self.ln2(edge_features)

        return semantic_features

def load_edge_features(file_path, d_model):
    with open(file_path, 'rb') as f:
        edge_features_dict = pickle.load(f)
        
    edge_features_list = list(edge_features_dict.values())
    if edge_features_list:
        edge_features_np = np.stack(edge_features_list)
    else:
        edge_features_np = np.array([])
        
    edge_features = torch.tensor(edge_features_np, dtype=torch.float)


    edge_features = torch.tensor(edge_features_np, dtype=torch.float)
    # Convert the dictionary to a PyTorch tensor
    # Adjust this conversion according to your data's format
    # Example for converting a simple list of feature vectors:
    edge_features = torch.tensor(list(edge_features_dict.values()), dtype=torch.float)

    # Ensure the shape is [num_edges, d_model]
    if edge_features.ndim == 1:
        edge_features = edge_features.view(-1, d_model)
    elif edge_features.ndim == 2 and edge_features.shape[1] != d_model:
        # Handle cases where the feature dimension is not d_model
        # Example: zero-padding or truncating features
        pass  # Implement as needed

    return edge_features

def process_directory(directory, d_model):
    processed_features = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            edge_features = load_edge_features(file_path, d_model)
            edge_features = edge_features.unsqueeze(1)  # Add batch dimension
            semantic_features = edge_attention_module(edge_features)
            processed_features.append(semantic_features)
    return processed_features

# Initialize the model
edge_attention_module = EdgeAttentionModule(d_model, num_heads)

output_directory = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features'
clean_directory = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\edge_features\clean'
vuln_directory = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\edge_features\vuln'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process and save features for each directory
clean_features = process_directory(clean_directory, d_model)
vuln_features = process_directory(vuln_directory, d_model)

# Save combined features
combined_features = (clean_features, vuln_features)
combined_features_file = os.path.join(output_directory, 'combined_semantic_features.pt')
torch.save(combined_features, combined_features_file)