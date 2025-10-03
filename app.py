import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


app = Flask(__name__)
CORS(app)

# Load the model architecture and weights
model = models.resnet18()  # Define base architecture
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # Adjust for 5 classes
model.load_state_dict(torch.load('diabetic_retinopathy_model.pth', map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set model to evaluation mode

# Define transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define class labels
class_labels = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR"}

@app.route('/')
def home():
    return "Diabetic Retinopathy Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0)  # Preprocess and add batch dimension

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    # Map to class label and return as JSON
    result = {"prediction": class_labels[predicted_class]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
