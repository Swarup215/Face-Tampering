from dataloader import get_dataloaders

train_loader, val_loader = get_dataloaders(batch_size=16)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))

for images, labels in train_loader:
    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    break