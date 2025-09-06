## 1. Overview of Project Files

This project implements three different classification algorithms —*Perceptron*, a *scratch-built neural network*, and a *PyTorch-based neural network*— to handle two different classification tasks. (1) classify handwritten digits (multi-class) or (2) classify face vs. non-face images (binary). 

Below is a brief summary of each file in the repository:

1. **`dataset.py`**  
   - Contains helper functions to load image data (digits or faces) from text files.  
   - Each dataset (train, validation, test) is contained in a pair of files: one with the ASCII-art images, another with the labels.  
   - The functions `load_digit_data` and `load_face_data` read the lines, parse the ASCII images into NumPy arrays, and return `(X_train, y_train), (X_val, y_val), (X_test, y_test)` tuples.  
   - Each image is flattened into a 1D vector

2. **`perceptron.py`**  
   - Contains `OneVsAllPerceptron` class, which trains one binary perceptron per class (i.e., a one-vs-rest strategy) for multi-class tasks.  
   - Each perceptron tracks a weight vector (and bias, if enabled) for each class.

3. **`nn_scratch.py`**  
   - Implements a **three-layer neural network** from scratch using NumPy.  
   - Uses ReLU activation in hidden layers and either softmax (multi-class) or sigmoid (binary) at the output.  
   - Performs forward propagation, computes cross-entropy loss (with optional L2 regularization), and uses manual backpropagation to update weights.

4. **`nn_pytorch.py`**  
   - Implements a **three-layer neural network** in **PyTorch**.  
   - Defines a `NeuralNetPyTorch` class with two hidden layers (ReLU activated) and one output layer.  
   - Provides a `train_pytorch_model` function to handle mini-batch training with Adam/SGD optimizer and the appropriate loss (`CrossEntropyLoss` for multi-class, `BCEWithLogitsLoss` for binary).

5. **`train.py`**  
   - This file is the main file to run the project !
   - Uses `argparse` to parse command-line arguments such as:  
     - `--task` (`digits` or `faces`)  
     - `--model` (`perceptron`, `nn_scratch`, `nn_pytorch`)  
     - `--epochs`, `--lr`, `--batch_size`, etc.  
   - Loads the corresponding dataset (`digits` or `faces`), normalizes the features, and sets the correct number of classes (10 for digits, 2 for faces).  
   - Trains the selected model on various fractions (0.1 to 1.0) of the training data multiple times to get average performance and standard deviation
   - Print results (training time, test accuracy, test error, test accuracy standard devication).  
   - These final results are also visualized with plots.

## 2.  How to Run `train.py`

1. **Ensure you have the required dependencies** (NumPy, PyTorch, Matplotlib, etc.):
   2.  **Navigate to the project directory**, where `train.py` is located.
   3. **Run `train.py`** with desired arguments. Examples:
```
# Example 1: Train the scratch-built neural network on the digit dataset:
python train.py --task digits --model nn_scratch --epochs 10 --lr 0.01

# Example 2: Train the one-vs-all perceptron on the face dataset, 
#            using 5 epochs and lr=0.005:
python train.py --task faces --model perceptron --epochs 5 --lr 0.005

# Example 3: Train the PyTorch neural network on the digit dataset with 20 epochs:
python train.py --task digits --model nn_pytorch --epochs 20

```

`train.py` will appropriate load either digit or face data, depending on `--task` and construct the specified model `--model`. It will train the model across a range of fractions of the training set (0.1, 0.2, …, 1.0), and print out the average training time and test accuracy across multiple runs per fraction.