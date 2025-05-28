import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load extracted features and labels for DenseNet121
features = np.load("/raid/home/posahemanth/Phase2/review2/Processed_Data/densenet121_featuresc.npy")
labels = np.load("/raid/home/posahemanth/Phase2/review2/Processed_Data/densenet121_labelsc.npy")
print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

# Convert labels to numerical format
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

# Convert to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(numeric_labels, dtype=torch.long)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_tensor.numpy())  # Convert to numpy for scaler
features = torch.tensor(features_scaled, dtype=torch.float32)

# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features, labels_tensor, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
print("Test labels distribution:", np.bincount(y_test.numpy()))
print("Train labels distribution:", np.bincount(y_train.numpy()))

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


# Simplified Vision Transformer for feature vector
# class VisionTransformerNet(nn.Module):
#     def __init__(self, input_size, num_classes, patch_size=100, dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1):
#         super(VisionTransformerNet, self).__init__()
        
#         num_patches = input_size // patch_size  # e.g., 3000 / 100 = 30
#         self.patch_size = patch_size
#         self.num_patches = num_patches
        
#         # Patch embedding
#         self.patch_embedding = nn.Linear(patch_size, dim)
        
#         # Positional embedding
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
#         # Transformer encoder (simplified)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(dim)
#         self.fc = nn.Linear(dim, num_classes)

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, self.num_patches, self.patch_size)
#         x = self.patch_embedding(x)
        
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embedding[:, :self.num_patches + 1]
#         x = self.dropout(x)
        
#         x = self.transformer(x)
#         x = self.norm(x[:, 0])
#         x = self.fc(x)
#         return x

# Hybrid CNN + Vision Transformer
# class CNNVisionTransformerNet(nn.Module):
#     def __init__(self, input_size, num_classes, feature_height=50, feature_width=60, patch_size=5, dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1):
#         super(CNNVisionTransformerNet, self).__init__()
        
#         # Input size must be compatible with feature_height * feature_width
#         assert input_size == feature_height * feature_width, f"input_size ({input_size}) must equal feature_height * feature_width ({feature_height * feature_width})"
        
#         self.feature_height = feature_height
#         self.feature_width = feature_width
#         self.patch_size = patch_size
        
#         # CNN Backbone
#         self.cnn = nn.Sequential(
#             # Input: [batch, 1, feature_height, feature_width] (e.g., [batch, 1, 50, 60])
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [batch, 16, 50, 60]
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [batch, 32, 50, 60]
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2),  # [batch, 32, 25, 30]
#         )
        
#         # Calculate patch dimensions after CNN
#         cnn_output_height = feature_height // 2  # After max pooling
#         cnn_output_width = feature_width // 2
#         self.num_patches = (cnn_output_height // patch_size) * (cnn_output_width // patch_size)  # e.g., (25/5) * (30/5) = 5 * 6 = 30 patches
#         patch_dim = 32 * patch_size * patch_size  # e.g., 32 * 5 * 5 = 800
        
#         # Patch embedding for ViT
#         self.patch_embedding = nn.Linear(patch_dim, dim)  # Project patches to dim (e.g., 800 → 128)
        
#         # Positional embedding
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(dim)
#         self.fc = nn.Linear(dim, num_classes)

#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # Reshape 1D features into 2D for CNN (e.g., [batch, 3000] → [batch, 1, 50, 60])
#         x = x.view(batch_size, 1, self.feature_height, self.feature_width)
        
#         # CNN forward pass
#         x = self.cnn(x)  # [batch, 32, 25, 30]
        
#         # Reshape for patching
#         batch_size, channels, height, width = x.size()
#         x = x.view(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [batch, h_patches, w_patches, channels, patch_size, patch_size]
#         x = x.view(batch_size, self.num_patches, channels * self.patch_size * self.patch_size)  # [batch, num_patches, patch_dim]
        
#         # Patch embedding
#         x = self.patch_embedding(x)  # [batch, num_patches, dim]
        
#         # Add CLS token
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)  # [batch, num_patches + 1, dim]
        
#         # Add positional embedding
#         x = x + self.pos_embedding[:, :self.num_patches + 1]
#         x = self.dropout(x)
        
#         # Transformer encoder
#         x = self.transformer(x)
#         x = self.norm(x[:, 0])  # CLS token
#         x = self.fc(x)
#         return x

# Initialize model
input_size = features.shape[1]  # Number of features (e.g., 3000 if FIXED_FEATURE_SIZE is unchanged)
num_classes = len(np.unique(labels))  # Number of unique classes

model = NeuralNet(input_size, num_classes)

# patch_size = 100  # Adjust based on input_size (must divide evenly, e.g., 3000 / 100 = 30 patches)

# model = VisionTransformerNet(
#     input_size=input_size,
#     num_classes=num_classes,
#     patch_size=patch_size,
#     dim=128,      # Reduced embedding size
#     depth=2,      # Fewer layers
#     heads=4,      # Fewer heads
#     mlp_dim=256   # Smaller feedforward
# )

# Initialize model
# input_size = 3000  # From your pipeline (FIXED_FEATURE_SIZE)
# num_classes = 4    # From your pipeline
# feature_height = 50
# feature_width = 60  # 50 * 60 = 3000
# patch_size = 5      # Patches from CNN output (25/5 * 30/5 = 30 patches)

# model = CNNVisionTransformerNet(
#     input_size=input_size,
#     num_classes=num_classes,
#     feature_height=feature_height,
#     feature_width=feature_width,
#     patch_size=patch_size,
#     dim=128,
#     depth=2,
#     heads=4,
#     mlp_dim=256,
#     dropout=0.1
# )

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize lists to store accuracy values
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    correct_train, total_train = 0, 0
    for features_batch, labels_batch in train_loader:
        features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels_batch).sum().item()
        total_train += labels_batch.size(0)

    train_accuracy = correct_train / total_train * 100
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for features_batch, labels_batch in val_loader:
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
            outputs = model(features_batch)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels_batch).sum().item()
            total_val += labels_batch.size(0)

    val_accuracy = correct_val / total_val * 100
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
    
print("Training Complete")
# Save trained model weights
torch.save(model.state_dict(), "/raid/home/posahemanth/Phase2/review2/Processed_Data/model_weightscnn.pth")
print("Model training complete. Weights saved to '/raid/home/posahemanth/Phase2/review2/Processed_Data/model_weightscnn.pth'.")

# Compute test accuracy
model.eval()
correct_test, total_test = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for features_batch, labels_batch in test_loader:
        features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
        outputs = model(features_batch)
        _, predicted = torch.max(outputs, 1)

        correct_test += (predicted == labels_batch).sum().item()
        total_test += labels_batch.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

test_accuracy = correct_test / total_test * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Load model weights (for verification)
model.load_state_dict(torch.load("/raid/home/posahemanth/Phase2/review2/Processed_Data/model_weightscnn.pth"))
model.to(device)
model.eval()
print("Model weights loaded successfully.")

# Print all accuracies
print(f"\nFinal Train Accuracies: {max(train_accuracies):.2f}%")
print(f"Final Validation Accuracies: {max(val_accuracies):.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.savefig("/raid/home/posahemanth/Phase2/review2/Processed_Data/accuracy_curvescnn.png")
# plt.show()

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
class_names = label_encoder.classes_  # Get class names

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("/raid/home/posahemanth/Phase2/review2/Processed_Data/confusion_matrixcnn.png")  # Save in Processed_Data
# plt.show()

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:\n", report)

# Save classification report as an image
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")
plt.text(0.01, 1, report, {'fontsize': 12}, fontproperties="monospace")
plt.savefig("/raid/home/posahemanth/Phase2/review2/Processed_Data/classification_reportcnn.png", bbox_inches="tight", dpi=300)  # Save in Processed_Data
# plt.show()

print("\nConfusion Matrix and Classification Report saved successfully!")