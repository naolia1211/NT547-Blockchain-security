import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class FeatureSelector(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(FeatureSelector, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.selector(x)

input_size_content = 1077
input_size_semantic = 130
output_size = 500

feature_selector_content = FeatureSelector(input_size_content, output_size)
feature_selector_semantic = FeatureSelector(input_size_semantic, output_size)

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
        
        label = semantic_features[-1]
        if isinstance(label, torch.Tensor) and label.numel() > 1:
            label = label[0]

        content_features = content_features.view(-1)[:input_size_content]
        semantic_features = semantic_features.view(-1)[:input_size_semantic]

        content_features = self.feature_selector_content(content_features.view(1, -1))
        semantic_features = self.feature_selector_semantic(semantic_features.view(1, -1))

        return content_features, semantic_features, label

class ClassificationModule(nn.Module):
    def __init__(self, reduced_features_dim, num_classes, dropout_rate=0.5):
        super(ClassificationModule, self).__init__()
        self.fc1 = nn.Linear(reduced_features_dim * 2, 1000)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, content_features, semantic_features):
        combined_features = torch.cat((content_features, semantic_features), dim=1)
        combined_features = F.relu(self.fc1(combined_features))
        combined_features = self.dropout1(combined_features)
        logits = self.fc2(combined_features)
        return logits

content_features_dir = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\content_features\pt_file'
semantic_features_dir = r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\features_extraction\asset\semantic_features\pt_file'

train_dataset = FeaturesDataset(content_features_dir, semantic_features_dir, feature_selector_content, feature_selector_semantic)
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

validation_dataset = FeaturesDataset(content_features_dir, semantic_features_dir, feature_selector_content, feature_selector_semantic)
validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

num_classes = 64
classifier = ClassificationModule(output_size, num_classes)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)


num_epochs = 30
total_steps = len(train_data_loader) * num_epochs
progress_bar = tqdm(total=total_steps, desc="Training Progress")

for epoch in range(num_epochs):
    classifier.train()
    
    for batch_idx, (content_features, semantic_features, labels) in enumerate(train_data_loader):
        optimizer.zero_grad()
        content_features = content_features.squeeze(1)
        semantic_features = semantic_features.squeeze(1)
        outputs = classifier(content_features, semantic_features)

        labels = labels.float().view(-1, num_classes)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item(), epoch=f"{epoch+1}/{num_epochs}")

progress_bar.close()


classifier.eval()
true_labels = []
predicted_labels = []

for content_features, semantic_features, labels in validation_data_loader:
    with torch.no_grad():
        content_features = content_features.squeeze(1)
        semantic_features = semantic_features.squeeze(1)
        
        outputs = classifier(content_features, semantic_features)
        predicted = torch.round(torch.sigmoid(outputs)).squeeze()

        labels = labels.squeeze(0)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

true_labels = [int(label) for label in true_labels]
predicted_labels = [int(pred) for pred in predicted_labels]

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(r'C:\Users\hao30\Documents\GitHub\NT547-Blockchain-security\Classification_module\confussion_matrix.png')
plt.close()  
