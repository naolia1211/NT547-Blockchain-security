import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Feature selector class
class FeatureSelector(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(FeatureSelector, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)  # Thêm lớp dropout ở đây
        )

    def forward(self, x):
        return self.selector(x)

# Feature sizes
input_size_content = 1185
input_size_semantic = 1183
output_size = 500

# Initialize feature selectors
feature_selector_content = FeatureSelector(input_size_content, output_size)
feature_selector_semantic = FeatureSelector(input_size_semantic, output_size)

# Dataset class
class FeaturesDataset(Dataset):
    def __init__(self, content_features_dir, semantic_features_dir, feature_selector_content, feature_selector_semantic):
        self.content_features_files = [os.path.join(content_features_dir, f) for f in os.listdir(content_features_dir)]
        self.semantic_features_files = [os.path.join(semantic_features_dir, f) for f in os.listdir(semantic_features_dir)]
        self.feature_selector_content = feature_selector_content
        self.feature_selector_semantic = feature_selector_semantic

    def __len__(self):
        return len(self.content_features_files)
    
    def __getitem__(self, idx):
        content_features = torch.load(self.content_features_files[idx])
        semantic_features = torch.load(self.semantic_features_files[idx])
        
        # Assuming the last element in semantic_features is the label
        label = semantic_features[-1]
        if isinstance(label, torch.Tensor) and label.numel() > 1:
            label = label[0]  # Taking the first element if it's a tensor with multiple elements

        content_features = content_features.view(-1)[:input_size_content]  # Reshape and truncate if necessary
        semantic_features = semantic_features.view(-1)[:input_size_semantic]  # Reshape and truncate if necessary

        content_features = self.feature_selector_content(content_features.view(1, -1))
        semantic_features = self.feature_selector_semantic(semantic_features.view(1, -1))

        return content_features, semantic_features, label
    
    
# Classification module
class ClassificationModule(nn.Module):
    def __init__(self, reduced_features_dim, dropout_rate=0.5):
        super(ClassificationModule, self).__init__()
        self.fc1 = nn.Linear(reduced_features_dim * 2, 1000)
        self.dropout1 = nn.Dropout(p=dropout_rate)  # Thêm lớp dropout
        self.fc2 = nn.Linear(1000, 100)

    def forward(self, content_features, semantic_features):
        combined_features = torch.cat((content_features, semantic_features), dim=1)
        combined_features = combined_features.view(-1, 1000)
        combined_features = F.relu(self.fc1(combined_features))
        combined_features = self.dropout1(combined_features)  # Áp dụng dropout sau kích hoạt
        logits = self.fc2(combined_features)
        return logits


# Directories for features
content_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\combine'
semantic_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features\combine'

# Initialize datasets and dataloaders
train_dataset = FeaturesDataset(content_features_dir, semantic_features_dir, feature_selector_content, feature_selector_semantic)
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Assume validation set is structured the same way as the training set
validation_dataset = FeaturesDataset(content_features_dir, semantic_features_dir, feature_selector_content, feature_selector_semantic)
validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Initialize model and optimizer
classifier = ClassificationModule(output_size)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 30  # Define number of epochs
for epoch in range(num_epochs):
    classifier.train()
    
    # Tạo một progress bar với tqdm
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (content_features, semantic_features, labels) in progress_bar:
        optimizer.zero_grad()

        # Reshape features for input to classifier
        content_features = content_features.squeeze(1)  # Removing extra dimensions
        semantic_features = semantic_features.squeeze(1)

        # Forward pass
        outputs = classifier(content_features, semantic_features)

        # Ensure labels are the correct shape
        labels = labels.float().view_as(outputs)  # Reshape labels to match output size

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Cập nhật progress bar với thông tin về loss
        progress_bar.set_postfix(loss=loss.item())
        
# Evaluation
classifier.eval()
true_labels = []
predicted_labels = []

for content_features, semantic_features, labels in validation_data_loader:
    with torch.no_grad():
        content_features = content_features.squeeze(1)
        semantic_features = semantic_features.squeeze(1)
        
        outputs = classifier(content_features, semantic_features)
        predicted = torch.round(torch.sigmoid(outputs)).squeeze()

        labels = labels.squeeze(0)  # Squeeze labels to one dimension

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

true_labels = [int(label) for label in true_labels]
predicted_labels = [int(pred) for pred in predicted_labels]

# Calculate and print metrics for multiclass classification
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Vẽ Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.savefig(r'D:\GitHub\NT547-Blockchain-security\Classification_module\confusion_matrix.png')
plt.close()  