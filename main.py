import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import resnet18
import warnings

warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = set()

        for img_name in os.listdir(root_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_name = img_name.rsplit('_', 1)[0]  # Extract class name before the underscore
                self.images.append(os.path.join(root_dir, img_name))
                self.labels.append(class_name)
                self.classes.add(class_name)

        self.classes = sorted(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros((3, 224, 224)), self.labels[idx]

class VisionModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionModel, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='mps'):
    print(f"Using device: {device}")
    model = model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        # Training
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")

def predict_image(model, image_path, transform, classes, device='mps'):
    """Predict the class of a single image."""
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            class_name = classes[predicted.item()]
            print(f"Predicted Class: {class_name}")
    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    start_time = time.time()
    torch.manual_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset_path = "Simpsons_Characters_Data/kaggle_simpson_testset/kaggle_simpson_testset/"
    print("Loading dataset...")
    dataset_start_time = time.time()
    full_dataset = CustomDataset(dataset_path, transform=train_transform)
    dataset_time = time.time() - dataset_start_time
    print(f"Loaded dataset with {len(full_dataset)} images in {dataset_time:.2f} seconds")

    # Split Dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply separate transforms for train/val
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Classes: {len(full_dataset.classes)} -> {full_dataset.classes}")

    # Model, Loss, Optimizer
    model = VisionModel(num_classes=len(full_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("Starting training...")
    train_start_time = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    train_time = time.time() - train_start_time
    print(f"Training completed in {train_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total script execution time: {total_time:.2f} seconds")

    # Predict an image
    print("\nModel training completed. Let's predict!")
    image_path = input("Enter the path of the image to predict: ")
    predict_image(model, image_path, val_transform, full_dataset.classes, device)

if __name__ == "__main__":
    main()
