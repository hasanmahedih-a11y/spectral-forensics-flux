import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os
import shutil
from codecarbon import EmissionsTracker

# Add project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA_DIR, ENERGY_LOG_DIR, ANALYSIS_RESOLUTION

# --- CONFIGURATION ---
BATCH_SIZE = 64  # Smaller batch size for CPU
EPOCHS = 3  # As you requested
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")  # Force CPU to measure the "pain" of Red AI


def train_baseline():
    print("\n" + "=" * 60)
    print(" STARTING BASELINE (ResNet-50) TRAINING")
    print(f"   - Hardware: CPU (This will be slow)")
    print(f"   - Epochs:   {EPOCHS}")
    print("=" * 60)

    # 1. SETUP DATA (Create a temp folder structure for PyTorch)
    # PyTorch ImageFolder requires: root/class_name/image.png
    base_dir = Path("data/temp_resnet_data")
    if base_dir.exists(): shutil.rmtree(base_dir)

    print("Preparing data layout...")
    for label_name, source_dir in [("real", RAW_DATA_DIR / "real"), ("fake", RAW_DATA_DIR / "flux1")]:
        dest_dir = base_dir / label_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # specific to your project: copy first 500 images to save time/space?
        # Or use all. Let's use ALL to make it a fair fight.
        for img_path in list(source_dir.glob("*.png")):
            shutil.copy(img_path, dest_dir / img_path.name)

    # 2. TRANSFORMS (Standard ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(base_dir, transform)

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data Loaded: {len(full_dataset)} images ({train_size} train / {test_size} test)")

    # 3. MODEL SETUP
    print("Loading ResNet-50 (Weights: ImageNet)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze layers (Fine-tuning)
    for param in model.parameters():
        param.requires_grad = False

    # Replace head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 Classes: Real vs Fake
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 4. TRAINING LOOP (With Energy Tracking)
    tracker = EmissionsTracker(output_dir=str(ENERGY_LOG_DIR), project_name="resnet_baseline_3epochs")
    tracker.start()

    print("\n--- Training Start ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{EPOCHS} in progress...")

        # Progress bar manually
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                print(f"   Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}", end="\r")

        epoch_acc = 100 * correct / total
        print(f"\n Epoch {epoch + 1} Complete. Train Acc: {epoch_acc:.2f}%")

    emissions = tracker.stop()

    # 5. FINAL EVALUATION
    print("\n--- Final Evaluation ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct / total
    print(f" Final ResNet Accuracy: {final_acc:.2f}%")
    print(f" Total Energy: {emissions:.5f} kWh")

    # Cleanup
    if base_dir.exists(): shutil.rmtree(base_dir)


if __name__ == "__main__":
    train_baseline()