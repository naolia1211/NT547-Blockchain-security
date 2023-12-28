import torch

# Đường dẫn tới file tensor
file_path = 'content_features.pt'
file_path = 'semantic_features.pt'

# Đọc tensor đã được lưu trước đó
content_features = torch.load(file_path)

# In ra tensor để kiểm tra
print(content_features)
