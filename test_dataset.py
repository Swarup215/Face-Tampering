from dataset import FaceDataset

real_ds = FaceDataset("faces/real", label=0)
fake_ds = FaceDataset("faces/fake", label=1)

print("Real images:", len(real_ds))
print("Fake images:", len(fake_ds))