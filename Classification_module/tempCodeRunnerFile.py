import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FeatureSelector(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureSelector, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.selector(x)

input_size_content = 1185  # Kích thước đầu vào cho content_features
input_size_semantic = 1183  # Kích thước đầu vào cho semantic_features
output_size = 500  # Kích thước đầu ra mong muốn

# Tạo hai instance của FeatureSelector, một cho mỗi loại đặc trưng
feature_selector_content = FeatureSelector(input_size_content, output_size)
feature_selector_semantic = FeatureSelector(input_size_semantic, output_size)

class FeaturesDataset(Dataset):
    def __init__(self, content_features_dir, semantic_features_dir, feature_selector):
        self.content_features_files = [os.path.join(content_features_dir, f) for f in os.listdir(content_features_dir)]
        self.semantic_features_files = [os.path.join(semantic_features_dir, f) for f in os.listdir(semantic_features_dir)]
        self.content_features_files.sort()
        self.semantic_features_files.sort()
        self.feature_selector = feature_selector

    def __len__(self):
        return len(self.content_features_files)

    def __getitem__(self, idx):
        content_features = torch.load(self.content_features_files[idx])
        semantic_features = torch.load(self.semantic_features_files[idx])
        
        # Áp dụng chọn lọc đặc trưng
        content_features = feature_selector_content(content_features.view(1, -1))
        semantic_features = feature_selector_semantic(semantic_features.view(1, -1))
        
        return content_features, semantic_features

class ClassificationModule(nn.Module):
    def __init__(self, reduced_features_dim, num_classes=2):
        super(ClassificationModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1, stride=1)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=9472)
        self.fc = nn.Linear(8 * 9472, num_classes)

    def forward(self, content_features, semantic_features):
        combined_features = torch.cat((content_features, semantic_features), dim=1)
        combined_features = combined_features.unsqueeze(1)
        conv1d_output = F.relu(self.conv1(combined_features))
        pooled_output = self.adaptive_pool(conv1d_output)
        flattened_output = pooled_output.view(1, -1)
        logits = self.fc(flattened_output)
        output = torch.sigmoid(logits)
        return output
    

# Khởi tạo DataLoader
content_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\vuln\Newfolder'  # Thay đổi đường dẫn tùy theo nhu cầu của bạn
semantic_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features\vuln'  # Thay đổi đường dẫn tùy theo nhu cầu của bạn
dataset = FeaturesDataset(content_features_dir, semantic_features_dir, feature_selector)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Khởi tạo và sử dụng mô hình ClassificationModule
classifier = ClassificationModule(output_size, num_classes=2)
classifier.eval()

# Xử lý dữ liệu và dự đoán
for content_features, semantic_features in data_loader:
    predictions = classifier(content_features, semantic_features)
    print(predictions)
