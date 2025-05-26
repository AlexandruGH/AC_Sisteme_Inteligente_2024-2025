import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Define class names
class_names = ['cat', 'dog']

class FineTunedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # <-- 2 classes: cat and dog
        )

    def forward(self, x):
        return self.base_model(x)


# Load fine-tuned ResNet18 model
@st.cache_resource
def load_model(weights_path):
    #model = models.resnet18(pretrained=False)
    #model.fc = nn.Linear(model.fc.in_features, 2)  # 2-class output
    model = FineTunedResNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🐶🐱 Dog vs. Cat Classifier")
st.write("Upload an image of a dog or a cat, and I’ll predict it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model("cat_dog_resnet18.pth")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(1).item()
        confidence = probs[0, pred_class].item()

    st.markdown(f"**Prediction:** {class_names[pred_class]} ({confidence:.2%} confidence)")