import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, label):
        self.samples = []
        self.label = label

        for video in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video)
            if not os.path.isdir(video_path):
                continue

            for img in os.listdir(video_path):
                self.samples.append(
                    os.path.join(video_path, img)
                )

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.label