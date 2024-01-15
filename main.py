from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import cv2
import numpy as np
import random
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
import pickle
from torchvision import transforms
# Using Googlenet for Category prediction
# Using edge detection and region growing to segment the object of interest and then extracts the dominant color from the segmented region
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pretrained category prediction model
category_model = torch.load("models/category_model.pth", map_location=device)
category_model.eval()
color_model = torch.load("models/color_model.pth", map_location=device)

categories = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items', 'Sporting Goods', 'Home']
colors = ['Black', 'White', 'Blue', 'Brown', 'Grey', 'Red', 'Green', 'Pink', 'Navy Blue', 'Purple', 'Other']

Others = ['Lavender', 'Grey Melange', 'Silver', 'Sea Green', 'Yellow', 'Rust', 'Magenta', 'Fluorescent Green', 'nan',
          'Turquoise Blue', 'Peach', 'Steel', 'Coffee Brown', 'Cream', 'Mustard', 'Nude', 'Off White', 'Beige', 'Teal',
          'Lime Green', 'Metallic', 'Bronze', 'Gold', 'Copper', 'Rose', 'Skin', 'Olive', 'Maroon', 'Orange',
          'Khaki', 'Charcoal', 'Tan', 'Taupe', 'Mauve', 'Burgundy', 'Mushroom Brown', 'Multi']
app = FastAPI()
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def segmentations(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the object of interest)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    # Extract the region of interest using the mask
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def extract_dominant_color(image):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_.astype(int)[0]
    return dominant_color

def extract_color(path):
    image = preprocess_image(path)
    roi = segmentations(np.array(image))
    dominant_color = extract_dominant_color(roi)
    
    subs = []
    for i in range(len(colors)):
        sums = 0
        for x in range(3):
            sums += abs(dominant_color[x])
        subs.append(sums)
    ind = subs.index(min(subs))
    if colors[ind] == "Other":
        return random.choice(Others)
    return colors[ind]

def predict_category(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = category_model(input_tensor)

    probabilities_category = torch.nn.functional.softmax(output[0], dim=0)
    predicted_category = categories[torch.argmax(probabilities_category).item()]
    return predicted_category

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    allowed_image_extensions = {"png", "jpg", "jpeg", "gif"}
    file_extension = file.filename.split(".")[-1]
    if file_extension.lower() not in allowed_image_extensions:
        raise HTTPException(status_code=400, detail="Invalid file format. Only images are allowed.")

    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    color = extract_color(file.filename)
    category = predict_category(file.filename)

    os.remove(file.filename)

    return {"Category": category, "Color": color}
