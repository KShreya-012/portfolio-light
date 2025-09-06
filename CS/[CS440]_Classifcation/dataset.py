
#dataset.py

#Utility functions for loading & preprocessing images into numpy arrays (or PyTorch tensors).
import os
import numpy as np


# General-purpose parsing helper functions
def _read_lines(file_path):
    """
    Reads a text file into a list of lines (strings), stripping newlines.
    """
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(line.rstrip('\n'))
    return lines

def _parse_as_images(lines, labels, num_rows, num_cols):
    """
    Given:
      - lines: a list of strings containing “ASCII art” for images
      - labels: a list of numeric labels (strings) with one label per image
      - num_rows, num_cols: height/width of each image

    Returns:
      X: shape (N, num_rows*num_cols), with 0/1 (or 0/255) features
      y: shape (N,)
    """
    N = len(labels)  # number of images
    X = []
    y = []
    
    # Each image occupies num_rows lines. We assume lines length = N * num_rows
    assert len(lines) == N * num_rows, (
        f"Mismatch: {len(lines)} lines vs. {N} images * {num_rows} rows per image."
    )
    
    idx_line = 0
    for i in range(N):
        # Collect the lines for this single image
        img_lines = lines[idx_line : idx_line + num_rows]
        idx_line += num_rows
        
        # Convert each pixel from ASCII to numeric
        # e.g. '#' or '+' means “on” pixel => 1, ' ' => 0 (for digits)
        # for faces, similarly interpret
        img_array = []
        for row_str in img_lines:
            # row_str has num_cols characters
            assert len(row_str) == num_cols, (
                f"Line length = {len(row_str)}, expected {num_cols}."
            )
            for char in row_str:
                if char == ' ':
                    img_array.append(0)
                else:
                    # If the pixel is '#' or '+', or something that indicates “on”
                    img_array.append(1)
        
        X.append(img_array)
        # Convert label from string to int
        y.append(int(labels[i]))
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)
    return X, y


# Digit data (28x28) loading functions


def load_digit_data(data_dir="data/digitdata"):
    """
    Loads digit data from the standard text files:
      - trainingimages (28x28) -> traininglabels
      - validationimages (28x28) -> validationlabels
      - testimages (28x28) -> testlabels

    Returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # 1) Read training set
    train_img_file = os.path.join(data_dir, "trainingimages")
    train_label_file = os.path.join(data_dir, "traininglabels")
    
    train_img_lines = _read_lines(train_img_file)
    train_labels = _read_lines(train_label_file)
    X_train, y_train = _parse_as_images(train_img_lines, train_labels, 28, 28)
    
    # 2) Read validation set
    val_img_file = os.path.join(data_dir, "validationimages")
    val_label_file = os.path.join(data_dir, "validationlabels")
    
    val_img_lines = _read_lines(val_img_file)
    val_labels = _read_lines(val_label_file)
    X_val, y_val = _parse_as_images(val_img_lines, val_labels, 28, 28)
    
    # 3) Read test set
    test_img_file = os.path.join(data_dir, "testimages")
    test_label_file = os.path.join(data_dir, "testlabels")
    
    test_img_lines = _read_lines(test_img_file)
    test_labels = _read_lines(test_label_file)
    X_test, y_test = _parse_as_images(test_img_lines, test_labels, 28, 28)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Face data (70x60) loading functions

def load_face_data(data_dir="data/facedata"):
    """
    Loads face data from the standard text files:
      - facedatatrain (70x60) -> facedatatrainlabels
      - facedatavalidation (70x60) -> facedatavalidationlabels
      - facedatatest (70x60) -> facedatatestlabels

    Returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # 1) Read training set
    train_img_file = os.path.join(data_dir, "facedatatrain")
    train_label_file = os.path.join(data_dir, "facedatatrainlabels")
    
    train_img_lines = _read_lines(train_img_file)
    train_labels = _read_lines(train_label_file)
    X_train, y_train = _parse_as_images(train_img_lines, train_labels, 70, 60)
    
    # 2) Read validation set
    val_img_file = os.path.join(data_dir, "facedatavalidation")
    val_label_file = os.path.join(data_dir, "facedatavalidationlabels")
    
    val_img_lines = _read_lines(val_img_file)
    val_labels = _read_lines(val_label_file)
    X_val, y_val = _parse_as_images(val_img_lines, val_labels, 70, 60)
    
    # 3) Read test set
    test_img_file = os.path.join(data_dir, "facedatatest")
    test_label_file = os.path.join(data_dir, "facedatatestlabels")
    
    test_img_lines = _read_lines(test_img_file)
    test_labels = _read_lines(test_label_file)
    X_test, y_test = _parse_as_images(test_img_lines, test_labels, 70, 60)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)