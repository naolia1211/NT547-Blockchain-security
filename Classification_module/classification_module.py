import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Feature selector class
class FeatureSelector(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureSelector, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
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
        
        # Extract the label. Ensure it is a single scalar value.
        label = semantic_features[-1]
        if isinstance(label, torch.Tensor):
            if label.numel() == 1:
                label = label.item()  # Convert tensor to a Python scalar
            else:
                raise ValueError(f"Label tensor has more than one element: {label.numel()} elements")

        semantic_features = semantic_features[:-1]  # Exclude the label

        content_features = content_features.view(1, -1)[:,:input_size_content]
        semantic_features = semantic_features.view(1, -1)[:,:input_size_semantic]

        content_features = self.feature_selector_content(content_features)
        semantic_features = self.feature_selector_semantic(semantic_features)

        return content_features, semantic_features, label
    
    
# Classification module
class ClassificationModule(nn.Module):
    def __init__(self, reduced_features_dim):
        super(ClassificationModule, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1, stride=1)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=9472)
        self.fc = nn.Linear(8 * 9472, 1)  # Output is 1 for binary classification

    def forward(self, content_features, semantic_features):
        combined_features = torch.cat((content_features, semantic_features), dim=1)
        combined_features = combined_features.view(1, 1, -1)
        conv1d_output = F.relu(self.conv1(combined_features))
        pooled_output = self.adaptive_pool(conv1d_output)
        flattened_output = pooled_output.view(1, -1)
        logits = self.fc(flattened_output)
        return logits

# Directories for features
content_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\vuln\Newfolder'
semantic_features_dir = r'D:\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features\vuln'

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
num_epochs = 10  # Define number of epochs
for epoch in range(num_epochs):
    classifier.train()
    for content_features, semantic_features, labels in train_data_loader:
        optimizer.zero_grad()
        outputs = classifier(content_features, semantic_features)
        labels = labels.float().unsqueeze(1)  # Ensure labels are correctly shaped
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
        
# Evaluation
classifier.eval()
true_labels = []
predicted_labels = []

for content_features, semantic_features, labels in validation_data_loader:
    with torch.no_grad():
        outputs = classifier(content_features, semantic_features)
        predicted = torch.round(torch.sigmoid(outputs)).squeeze()
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Calculate and print metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='binary')
recall = recall_score(true_labels, predicted_labels, average='binary')
f1 = f1_score(true_labels, predicted_labels, average='binary')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
