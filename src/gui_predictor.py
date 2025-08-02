import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
import network  # your own file
import pickle
from scipy import ndimage

# Load trained model
net = network.Network([1024, 30, 10])
net.load('digit_model.npz')

# Create GUI window
window = tk.Tk()
window.title("Digit Recognizer")

canvas_width = 320
canvas_height = 320
canvas_bg = "white"

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg=canvas_bg)
canvas.pack()

image1 = Image.new("L", (canvas_width, canvas_height), color=255)
draw = ImageDraw.Draw(image1)

# Drawing logic
def draw_callback(event):
    x, y = event.x, event.y
    r = 12
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

canvas.bind("<B1-Motion>", draw_callback)

# Clear canvas
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=255)
    result_label.config(text="")

# Predict digit

def predict():
    img = ImageOps.invert(image1.copy()).convert('L')
    #img = img.point(lambda x: 255 if x > 20 else 0, mode='1')  # simple threshold
    img = img.resize((32, 32))
    #img = img.point(lambda x: 255 if x > 20 else 0, mode='1')  # simple threshold
    #img = img.filter(ImageFilter.GaussianBlur(radius=0))
    img.save("canvas_digit.png")
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_flat = img_array.reshape(1024, 1)
    
    output = net.feedforward(img_flat)
    prediction = np.argmax(output)
    
    result_label.config(text=f"Prediction: {prediction}")
    


# Buttons
button_frame = tk.Frame(window)
button_frame.pack()

tk.Button(button_frame, text="Predict", command=predict).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Clear", command=clear).pack(side=tk.LEFT, padx=10)

# Result label
result_label = tk.Label(window, text="", font=("Helvetica", 20))
result_label.pack(pady=10)

window.mainloop()

