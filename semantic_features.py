import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# Khởi tạo mô hình
edge_attention_module = EdgeAttentionModule(d_model, num_heads)

# Giả sử edge_features là tensor của bạn với shape [num_edges, d_model]
# Ví dụ: edge_features = torch.rand((edge_indices.size(0), d_model))  # Thay thế bằng dữ liệu thực của bạn
edge_features = torch.rand((edge_indices.size(0), 1, d_model))  # Thêm một chiều batch

# Áp dụng mô hình lên các edge features
semantic_features = edge_attention_module(edge_features)

# Lưu semantic_features vào file .pt
torch.save(semantic_features, 'semantic_features.pt')
