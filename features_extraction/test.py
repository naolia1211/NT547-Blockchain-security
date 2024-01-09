import torch
import os

# Replace with the actual path to your directories containing .pt files
content_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\vuln\Newfolder'
semantic_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features\vuln'

# Get the first .pt file from each directory
content_features_file = next((f for f in os.listdir(content_features_dir) if f.endswith('.pt')), None)
semantic_features_file = next((f for f in os.listdir(semantic_features_dir) if f.endswith('.pt')), None)

# Check if files were found
if content_features_file is None or semantic_features_file is None:
    raise FileNotFoundError("Could not find .pt files in the specified directories.")

# Load the tensors from the .pt files
content_features_path = os.path.join(content_features_dir, content_features_file)
semantic_features_path = os.path.join(semantic_features_dir, semantic_features_file)

content_features = torch.load(content_features_path)
semantic_features = torch.load(semantic_features_path)

# Get the dimensions of the content and semantic features
content_features_dim = content_features.size(0)  # Assuming the features are in the first dimension
semantic_features_dim = semantic_features.size(0)  # Assuming the features are in the first dimension

print(f'Content Features Dimension: {content_features_dim}')
print(f'Semantic Features Dimension: {semantic_features_dim}')
