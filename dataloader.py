from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import FaceDataset

def get_dataloaders(batch_size=16):
    # Load datasets
    real_ds = FaceDataset("faces/real", label=0)
    fake_ds = FaceDataset("faces/fake", label=1)

    # Combine
    full_dataset = ConcatDataset([real_ds, fake_ds])

    # Split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader