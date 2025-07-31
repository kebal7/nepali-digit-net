import os
import numpy as np
from PIL import Image
from network import Network

# --- Preprocess image ---
def process_image(file_path):
    img = Image.open(file_path).convert('L')       # Grayscale
    img = img.resize((32, 32))                     # Resize to MNIST size
    img_array = np.array(img)
    #img_array = 255 - img_array                    # Invert: black bg, white digit
    img_array = img_array / 255.0                  # Normalize to [0, 1]
    img_flat = img_array.reshape(1024, 1)           # Flatten to 1024x1
    return img_flat

# --- Load model ---
net = Network([1024, 30, 10])
net.load('digit_model.npz')  # Loads from model/ directory
print("✅ Model loaded.")

# --- Set image path inside data/test ---
image_name = 'digit_test.png'  # You can change this filename
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'test', 'digit.png'))

# --- Check if image exists ---
if not os.path.exists(image_path):
    print(f"❌ Image not found at {image_path}")
    exit()

# --- Load and predict ---
x = process_image(image_path)
output = net.feedforward(x)
prediction = np.argmax(output)

print("🧠 Predicted digit:", prediction)

