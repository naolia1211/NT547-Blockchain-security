import torch

# Đường dẫn tới file tensor
file_path = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\semantic_features.pt'


# Đọc tensor đã được lưu trước đó
content_features = torch.load(file_path)

# In ra tensor để kiểm tra
print(content_features)
