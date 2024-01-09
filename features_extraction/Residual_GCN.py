import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import numpy
import os


class ResidualGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        # Định nghĩa các lớp biến đổi để kích thước phù hợp với output của mỗi GCNConv
        self.transform1 = torch.nn.Linear(in_channels, 16)
        self.transform2 = torch.nn.Linear(16, out_channels)

    def forward(self, x, edge_index):
        # Sử dụng lớp biến đổi trước khi thực hiện phép cộng
        identity1 = self.transform1(x)
        x = F.relu(self.conv1(x, edge_index)) + identity1
        # Biến đổi x để có kích thước phù hợp với output của conv2
        identity2 = self.transform2(x)
        x = F.relu(self.conv2(x, edge_index)) + identity2
        return x

def load_and_process(data_directory, filename, output_directory):
    # Load node features từ file pickle
    file_path = os.path.join(data_directory, filename)
    with open(file_path, 'rb') as f:
        node_features = pickle.load(f)

    node_features_tensor = torch.tensor(numpy.array([node_features[node] for node in sorted(node_features.keys())]), dtype=torch.float)

    # Ví dụ: Định nghĩa edge_index
    edge_index_data = [(0, 1), (1, 2), (2, 3)]
    edge_index = torch.tensor(edge_index_data, dtype=torch.long).t().contiguous()
    desired_output_dim = 64
    gcn_model = ResidualGCN(in_channels=node_features_tensor.shape[1], out_channels=desired_output_dim)
    
    
    processed_features = gcn_model(node_features_tensor, edge_index)

    # Lưu kết quả sau khi thực hiện forward pass
    output_filename = os.path.splitext(filename)[0] + '.pt'
    output_path = os.path.join(output_directory, output_filename)
    torch.save(processed_features, output_path)


def process_directory(data_directory, output_directory):
    for filename in os.listdir(data_directory):
        if filename.endswith('.pkl'):
            load_and_process(data_directory, filename, output_directory)


output_directory_vuln = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\vuln'
output_directory_clean = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\clean'

# Process 'clean' directory
clean_data_directory = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\node_features\clean'
process_directory(clean_data_directory, output_directory_clean)

# Process 'vuln' directory
vuln_data_directory = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\node_features\vuln'
process_directory(vuln_data_directory, output_directory_vuln)