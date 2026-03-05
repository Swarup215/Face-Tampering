import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from mesonet import MesoNet
from dataloader import get_dataloaders

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- DATA ----------------
train_loader, val_loader = get_dataloaders(batch_size=32)

# ---------------- MODEL ----------------
model = MesoNet().to(device)

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------- TRAINING SETTINGS ----------------
epochs = 10

# ---------------- TRAIN LOOP ----------------
for epoch in range(epochs):
    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    # -------- TRAIN --------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in train_bar:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100*correct/total:.2f}%"
        )

    train_acc = 100 * correct / total
    import os

    os.makedirs("model", exist_ok=True)

    torch.save(model.state_dict(), "model/model.pth")
    print("✅ Model saved successfully to model/model.pth")
    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    val_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = (outputs >= 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    val_acc = 100 * val_correct / val_total

    # -------- EPOCH SUMMARY --------
    print(
        f"Epoch {epoch+1}: "
        f"Train Acc = {train_acc:.2f}% | "
        f"Val Acc = {val_acc:.2f}%"
    )

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "mesonet_ffpp.pth")
print("\nTraining complete. Model saved as mesonet_ffpp.pth")