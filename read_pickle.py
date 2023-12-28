import pickle

# Đường dẫn đến file Pickle
file_path = 'node_features.pkl'
file_path = 'edge_features.pkl'

# Mở và đọc file Pickle
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# In ra nội dung để kiểm tra
print(data)
