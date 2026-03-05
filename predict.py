import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ======================================================
# EXACT MODEL — MUST MATCH TRAINING ARCHITECTURE
# ======================================================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # Input: 128x128 → after 4 pools → 8x8
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


# ======================================================
# LOAD MODEL
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)
state = torch.load("model/model.pth", map_location=device)
model.load_state_dict(state,strict=False)
model.eval()

# ======================================================
# IMAGE TRANSFORM (MUST MATCH TRAINING)
# ======================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ======================================================
# PREDICTION FUNCTION
# ======================================================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(img).item()

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if label == "FAKE" else 1 - prob

    return label, round(confidence, 4)