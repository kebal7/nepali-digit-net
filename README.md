# Nepali Handwritten Digit Recognition

This project implements a neural network **from scratch** using NumPy to recognize Nepali handwritten digits (0–9). The model is trained on a Kaggle dataset of Nepali digits, offering a specialized alternative to MNIST.

## 🧠 Features

- **Custom Neural Network**: Built without external ML libraries, only NumPy.
- **Nepali Digit Dataset**: Trained on Kaggle's Nepali handwritten digits.
- **Training & Prediction**:
  - Mini-batch stochastic gradient descent training (`src/train.py`)
  - Testing on images (`src/test_image.py`)
  - GUI predictor to draw digits and predict interactively (`gui_predictor/`)
- **Model Storage**: Saved and loaded from the `model/` directory.
- **Performance**: Over 90% accuracy on test set.

## 📦 Requirements

- Python 3.x
- NumPy
- OpenCV (for GUI predictor)

Install dependencies:

```bash
pip install numpy opencv-python
```

## 🚀 Usage

### Clone Repository

```bash
git clone https://github.com/kebal7/nepali-digit-net.git
cd nepali-digit-net
```

### Prepare Dataset

Download the Nepali handwritten digit dataset from Kaggle and organize it under the `data/` folder as per the existing structure.

### Train Model

```bash
python src/main.py
```

Model parameters and weights will be saved automatically in the `model/` folder.

### Predict from Test Images

Place your test images inside `data/test/` and run:

```bash
python src/test_image.py
```

This will load the saved model and print predictions for each image.

### Use GUI Predictor

To draw digits interactively and get predictions, run:

```bash
python src/gui_predictor.py
```

Use your mouse to draw a digit and click the **Predict** button to see the model's prediction instantly.

## 📂 Project Structure

```
nepali-digit-net/
├── data/
│   ├── train/           # Training images(in csv) dataset
│   ├── test/            # Test images(in csv) for prediction
├── model/               # Saved neural network weights and parameters
├── src/
│   ├── main.py         # Script to train and save model
│   ├── test_image.py    # Script to test on new images
│   ├── network.py       # Neural network implementation
│   ├── gui_predictor.py # GUI with canvas to draw and predict
│   └── load              # Loads data from csv
└── README.md            # This documentation
```

## 📈 Sample Training Output

```
Epoch 0: 8423 / 10000
Epoch 1: 8762 / 10000
...
Epoch 29: 9065 / 10000
```


# nepali-digit-net
