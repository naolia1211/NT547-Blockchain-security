import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa module phân loại
class ClassificationModule(nn.Module):
    def __init__(self, combined_features_dim, num_classes):
        super(ClassificationModule, self).__init__()
        self.combined_features_dim = combined_features_dim

        # Khởi tạo các lớp mạng
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=self.combined_features_dim, stride=self.combined_features_dim)
        self.fc = nn.Linear(2, num_classes)  # Đầu ra có 2 features từ Conv1d
    
    def forward(self, combined_features):
        # Xử lý tensor đầu vào
        combined_features = combined_features.view(1, 1, -1)
        conv1d_output = self.conv1d(combined_features)
        pooled_output, _ = torch.max(conv1d_output, dim=2)
        flattened_output = pooled_output.view(-1)
        classification_output = self.fc(flattened_output.unsqueeze(0))
        return torch.sigmoid(classification_output)
        
        # Áp dụng Conv1d
        conv1d_output = self.conv1d(combined_features)
        
        # Loại bỏ các chiều không cần thiết và flatten output
        conv1d_output = conv1d_output.view(-1)
        
        # Áp dụng Linear layer để nhận kết quả phân loại
        classification_output = self.fc(conv1d_output.unsqueeze(0))
        return torch.sigmoid(classification_output)

# Load features
content_features = torch.load('content_features.pt')
semantic_features = torch.load('semantic_features.pt')

# Số lượng features từ content và semantic features
num_features = content_features.shape[1] + semantic_features.shape[1]

# Khởi tạo module phân loại với 1 là số lớp (đối với phân loại nhị phân)
classifier = ClassificationModule(num_features, 1)

# Chuyển module vào chế độ đánh giá
classifier.eval()

# Định nghĩa biến để lưu kết quả dự đoán
prediction = None

with torch.no_grad():
    # Đảm bảo cả hai tensor cùng thiết bị và không yêu cầu gradient
    content_features = content_features.to('cpu').detach()
    semantic_features = semantic_features.detach()

    # Định dạng lại semantic_features để có cùng số lượng chiều với content_features
    # Chú ý rằng ở đây chúng ta giả định rằng semantic_features có ba chiều và chúng ta muốn loại bỏ chiều batch đầu tiên
    semantic_features = semantic_features.squeeze(0)  # Loại bỏ chiều batch, giả sử kích thước ban đầu là [1, K, C]

    # Làm phẳng semantic_features để có cùng kích thước với content_features
    # Giả sử content_features là [K, ∑ct] và semantic_features là [K, C] sau khi loại bỏ chiều batch
    semantic_features_flat = semantic_features.view(-1)  # Làm phẳng thành [K * C]

    # Đảm bảo content_features là một vector phẳng [K * ∑ct]
    content_features_flat = content_features.view(-1)  # Làm phẳng thành [K * ∑ct]

    # Nối chúng lại với nhau để tạo thành [1, K * (∑ct + C)]
    combined_features = torch.cat((content_features_flat, semantic_features_flat), dim=0).unsqueeze(0)

    # Thực hiện dự đoán
    prediction = classifier(combined_features)

# Lấy xác suất từ tensor đầu ra
probability = prediction.item()  # Đây là giá trị xác suất từ 0 đến 1

# Phân loại dựa trên ngưỡng
classification = "vulnerable" if probability >= 0.5 else "clean"

# In giá trị xác suất và phân loại
print(f'Probability of being vulnerable: {probability:.4f}')
print(f'The function is classified as: {classification}')



