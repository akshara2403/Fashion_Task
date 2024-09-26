# from fastapi import FastAPI, File, UploadFile, HTTPException

# from torchvision import transforms
# import torchvision.transforms as transforms
# from PIL import Image
# from sklearn.cluster import KMeans
# from io import BytesIO
# import numpy as np
# import random
# import cv2
# import pickle
# import os
# app = FastAPI()

# # Load pre-trained ResNet model for category prediction
# category_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# category_model.eval()

# # Color encoding dictionary with RGB values
# color_encoding = {
#     'Navy Blue': (0, 0, 128),
#     'Blue': (0, 0, 255),
#     'Silver': (192, 192, 192),
#     'Black': (0, 0, 0),
#     'Grey': (128, 128, 128),
#     'Green': (0, 128, 0),
#     'Purple': (128, 0, 128),
#     'White': (255, 255, 255),
#     'Beige': (245, 245, 220),
#     'Brown': (165, 42, 42),
#     'Bronze': (205, 127, 50),
#     'Teal': (0, 128, 128),
#     'Copper': (184, 115, 51),
#     'Pink': (255, 192, 203),
#     'Off White': (255, 255, 224),
#     'Maroon': (128, 0, 0),
#     'Red': (255, 0, 0),
#     'Khaki': (165, 42, 42),
#     'Orange': (255, 165, 0),
#     'Yellow': (255, 255, 0),
#     'Charcoal': (54, 69, 79),
#     'Gold': (255, 223, 0),
#     'Steel': (79, 82, 89),
#     'Tan': (210, 180, 140),
#     'Multi': (255, 0, 255),
#     'Magenta': (255, 0, 255),
#     'Lavender': (230, 230, 250),
#     'Sea Green': (46, 139, 87),
#     'Cream': (255, 253, 208),
#     'Peach': (255, 218, 185),
#     'Olive': (128, 128, 0),
#     'Skin': (255, 182, 193),
#     'Burgundy': (128, 0, 0),
#     'Coffee Brown': (111, 78, 55),
#     'Grey Melange': (169, 169, 169),
#     'Rust': (183, 65, 14),
#     'Rose': (233, 150, 122),
#     'Lime Green': (50, 205, 50),
#     'Mauve': (224, 176, 255),
#     'Turquoise Blue': (64, 224, 208),
#     'Metallic': (169, 169, 169),
#     'Mustard': (255, 255, 0),
#     'Taupe': (139, 133, 137),
#     'Nude': (255, 222, 173),
#     'Mushroom Brown': (153, 101, 21),
#     'Fluorescent Green': (0, 255, 0)
# }

# # Define other categories and colors
# categories = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home']
# colors = list(color_encoding.keys())
# others = ['Other']

# def preprocess_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = transform(image)
#     input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
#     return input_batch

# def extract_dominant_color(image_path, k=1):
#     image = Image.open(image_path).convert('RGB')
#     pixels = np.array(image).reshape((-1, 3))
#     kmeans = KMeans(n_clusters=k, n_init=10)
#     kmeans.fit(pixels)
#     dominant_color = kmeans.cluster_centers_.astype(int)
#     return dominant_color[0]

# def match_color_to_encoding(dominant_color):
#     closest_color_label = min(color_encoding, key=lambda x: np.linalg.norm(np.array(dominant_color) - np.array(color_encoding[x])))
#     return closest_color_label

# def predict_category(image_path):
#     input_tensor = preprocess_image(image_path)
#     output = category_model(input_tensor)
#     _, predicted_idx = torch.max(output, 1)
#     return categories[predicted_idx.item()]

# # Example: Upload image and predict category and color
# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     # Ensure that the file is an image (you can customize this check based on your requirements)
#     allowed_image_extensions = {"png", "jpg", "jpeg", "gif"}
#     file_extension = file.filename.split(".")[-1]
#     if file_extension.lower() not in allowed_image_extensions:
#         raise HTTPException(status_code=400, detail="Invalid file format. Only images are allowed.")

#     # Save the uploaded file (you might want to customize the path)
#     with open(file.filename, "wb") as f:
#         f.write(file.file.read())

#     # Predict category
#     predicted_category = predict_category(file.filename)

#     # Extract dominant color
#     dominant_color = extract_dominant_color(file.filename)

#     # Match color to encoding
#     predicted_color = match_color_to_encoding(dominant_color)

#     os.remove(file.filename)

#     return {"Predicted Category": predicted_category, "Predicted Color": predicted_color}
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category_model = torch.load("googlenet_category_model.pth", map_location=device)
color_model = torch.load("color_model.pth", map_location=device)
category_model.eval()
color_model.eval()

categories = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home']
colors = ['Black', 'White', 'Blue', 'Brown', 'Grey', 'Red', 'Green', 'Pink', 'Navy Blue', 'Purple']

Others = ['Lavender', 'Grey Melange', 'Silver', 'Sea Green', 'Yellow', 'Rust', 'Magenta', 'Fluorescent Green', 'nan',
          'Turquoise Blue', 'Peach', 'Steel', 'Coffee Brown', 'Cream', 'Mustard', 'Nude', 'Off White', 'Beige', 'Teal',
          'Lime Green', 'Metallic', 'Bronze', 'Gold', 'Copper', 'Rose', 'Skin', 'Olive', 'Maroon', 'Orange',
          'Khaki', 'Charcoal', 'Tan', 'Taupe', 'Mauve', 'Burgundy', 'Mushroom Brown', 'Multi']

app = FastAPI()

# Define the image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Ensure that the file is an image (you can customize this check based on your requirements)
    allowed_image_extensions = {"png", "jpg", "jpeg", "gif"}
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() not in allowed_image_extensions:
        raise HTTPException(status_code=400, detail="Invalid file format. Only images are allowed.")

    # # Save the uploaded file (you might want to customize the path)
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    input_tensor = preprocess_image(file.filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output1 = category_model(input_tensor)
        output2 = color_model(input_tensor)
    
    probabilities_category = torch.nn.functional.softmax(output1[0], dim=0)
    probabilities_color = torch.nn.functional.softmax(output2[0], dim=0)

    category = categories[torch.argmax(probabilities_category)]
    color_ind = torch.argmax(probabilities_color)
    color = ""
    if color_ind == len(colors):
        color = random.choice(Others)
    else:
        color = colors[color_ind]

    os.remove(file.filename)

    return {"Category": category, 
            "Color" : color}
