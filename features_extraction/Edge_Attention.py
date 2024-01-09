import torch
import torch.nn as nn
import math
import pickle
import numpy as np
import os


# Xác định thiết bị mà mô hình sẽ chạy trên: GPU nếu có sẵn, ngược lại là CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa mô hình EdgeAttentionModule
class EdgeAttentionModule(nn.Module):
    def __init__(self, d_model=64, num_heads=4):  # Giảm d_model xuống còn 64
        super(EdgeAttentionModule, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),  # Tối ưu hóa ReLU
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)


    def forward(self, edge_features):
        # Positional Encoding cho edge_features
        seq_length, batch_size = edge_features.size(0), edge_features.size(1)
        position = torch.arange(seq_length, dtype=torch.float, device=edge_features.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_length, self.d_model, device=edge_features.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1).repeat(1, batch_size, 1)
        edge_features = edge_features + pe
        
        # Multi-head attention
        edge_features = edge_features.permute(1, 0, 2)  # Chuyển vị để phù hợp với kỳ vọng của MultiheadAttention
        attn_output, _ = self.mha(edge_features, edge_features, edge_features)
        attn_output = attn_output.permute(1, 0, 2)  # Chuyển vị lại sau khi attention
        # Add & Norm
        edge_features = edge_features + attn_output
        edge_features = self.layer_norm1(edge_features)
        
        # Feed Forward Network
        ffn_output = self.ffn(edge_features)
        # Add & Norm
        semantic_features = edge_features + ffn_output
        semantic_features = self.layer_norm2(semantic_features)

        return semantic_features

# Hàm tải dữ liệu từ file .pkl
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Hàm lưu dữ liệu ra file .pt
def save_tensor_to_pt(tensor, file_path):
    torch.save(tensor, file_path)

# Hàm xử lý dữ liệu từng file .pkl qua mô hình và lưu kết quả ra file .pt
def process_file(file_path, module, d_model=64):
    # Load dữ liệu từ file .pkl
    data_dict = load_pickle(file_path)
    
    # Chuyển đổi dữ liệu thành numpy array
    edge_features_list = list(data_dict.values())
    edge_features_np = np.stack(edge_features_list) if edge_features_list else np.array([], dtype=np.float32)

    # Tính kích thước tổng cộng của tensor sau khi thêm padding
    total_elements = edge_features_np.shape[0] * d_model
    required_elements = (64 - total_elements % 64) % 64
    required_rows = required_elements // d_model

    # Thêm padding hoặc cắt bớt dữ liệu
    if required_rows > 0:
        padding = np.zeros((required_rows, edge_features_np.shape[1]))
        edge_features_np = np.concatenate((edge_features_np, padding), axis=0)

    # Chuyển đổi dữ liệu thành tensor và định dạng lại cho phù hợp
    edge_features_tensor = torch.tensor(edge_features_np, dtype=torch.float).reshape(-1, 1, d_model).to(device)
    
    # Xử lý qua mô hình EdgeAttentionModule
    semantic_features = module(edge_features_tensor)
    
    # Lưu kết quả ra file .pt
    output_file_path = os.path.join(output_directory, os.path.splitext(os.path.basename(file_path))[0] + '_semantic.pt')
    save_tensor_to_pt(semantic_features.detach(), output_file_path)



# Khởi tạo mô hình và đặt nó trên thiết bị đã chọn
edge_attention_module = EdgeAttentionModule().to(device)  # Sử dụng d_model mặc định là 64

# Đường dẫn thư mục chứa các file .pkl và nơi lưu kết quả
input_directory = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\edge_features\vuln'
output_directory = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features\vuln1'

# Đảm bảo rằng thư mục đầu ra tồn tại
os.makedirs(output_directory, exist_ok=True)


# Xử lý tất cả các file .pkl trong thư mục đầu vào
for filename in os.listdir(input_directory):
    if filename.endswith('.pkl'):
        file_path = os.path.join(input_directory, filename)
        process_file(file_path, edge_attention_module, d_model=64)
