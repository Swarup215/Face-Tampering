import torch
from torchvision import transforms
from PIL import Image
from mesonet import MesoNet

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = MesoNet().to(device)
model.load_state_dict(torch.load("mesonet_ffpp.pth", map_location=device))
model.eval()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 🔴 CHANGE THIS PATH to any face image
img_path = "C:\\Users\\rishi\\Downloads\\Gemini_Generated_Image_bzdpkjbzdpkjbzdp.png"

img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# ---------------- PREDICTION ----------------
with torch.no_grad():
    output = model(img)
    prob = output.item()

label = "FAKE" if prob >= 0.5 else "REAL"

print(f"Prediction: {label}")
print(f"Confidence: {prob:.4f}")