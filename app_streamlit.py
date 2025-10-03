import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Load Model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("diabetic_retinopathy_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_labels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Streamlit UI
st.title("👁️ Diabetic Retinopathy Detection")
st.write("Upload a retinal image to classify DR stage")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Prediction: **{class_labels[predicted.item()]}**")
