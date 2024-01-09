import torch

# Đường dẫn tới file tensor
file_path = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\clean\0b71bee14c6fb53ac5d452e429d57f5020106d3e_ZeroBTCWorldCup_node_features.pt'


# Đọc tensor đã được lưu trước đó
content_features = torch.load(file_path)

# In ra tensor để kiểm tra
print(content_features)
