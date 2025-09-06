#Orchestrates the training loops, command-line arguments, experiment runs, etc

#1. Load data (digits, faces).
#2. Initialize whichever model you want (perceptron, scratch NN, or PyTorch NN).
#3. Train it with the specified fraction of training data.
#4. Evaluate on the test set.
#5. Print or log results.

### handle command-line arguments (using argparse) to pick which classifier and how much training data to use

# train.py
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt

from dataset import load_digit_data, load_face_data
from perceptron import OneVsAllPerceptron
from nn_scratch import NeuralNetScratch
from nn_pytorch import NeuralNetPyTorch, train_pytorch_model

def evaluate_accuracy(model, X, y):
    """Compute classification accuracy on (X, y)."""
    y_pred = model.predict(X)
    return np.mean(y_pred == y)

def main():
    parser = argparse.ArgumentParser(description="Train script reflecting nn_scratch softmax/sigmoid changes.")
    parser.add_argument('--task', type=str, default='digits',
                        choices=['digits','faces'],
                        help="Which dataset to load (digits or faces).")
    parser.add_argument('--model', type=str, default='nn_scratch',
                        choices=['perceptron','nn_scratch','nn_pytorch'],
                        help="Which model to train.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size (for nn_pytorch).")
    args = parser.parse_args()

    # runs = 3 for standard deviation calculations:
    repeats = 3

    # 1) Load the dataset
    if args.task == 'digits':
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_digit_data()
        """
        EXPERIMENT
        """
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-7
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        # Now X_train, X_val, X_test are standardized per feature
        """
        """

        num_classes = 10  # multi-class
        # Possibly smaller hidden layers for digits
        hidden_dim1, hidden_dim2 = 64, 32
    else:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_face_data()

        """
        EXPERIMENT
        """
        # 3) Compute per-feature (pixel) mean & std on the TRAIN set
        mean = np.mean(X_train, axis=0)       # shape: (num_features,)
        std = np.std(X_train, axis=0) + 1e-7  # add small epsilon to avoid division by zero

        # 4) Normalize train, val, test with that mean & std
        X_train = (X_train - mean) / std
        X_val   = (X_val - mean) / std
        X_test  = (X_test - mean) / std
        # Now X_train, X_val, X_test are standardized per feature
        """
        """

        num_classes = 2   # face vs non-face => binary
        # Often faces need bigger layers for better accuracy
        hidden_dim1, hidden_dim2 = 128, 64

    input_dim = X_train.shape[1]
    print(f"[INFO] Loaded {args.task.upper()} data")
    print(f"  Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val set:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test set:  X={X_test.shape}, y={y_test.shape}")

    # Fractions of training data to try
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for frac in fractions:
        N = int(len(X_train) * frac)
        run_times = []
        run_accuracies = []

        for _ in range(repeats):
            # Random subset of the training data
            indices = np.random.choice(len(X_train), size=N, replace=False)
            X_sub = X_train[indices]
            y_sub = y_train[indices]

            #############################sc
            # Initialize the model
            #############################
            if args.model == 'nn_scratch':
                from nn_scratch import NeuralNetScratch

                # If multi-class => output_dim = 10
                # If binary => output_dim = 1
                if num_classes > 2:
                    output_dim = num_classes  # digits => softmax
                else:
                    output_dim = 1  # faces => sigmoid

                # Construct the network (ReLU hidden, softmax or sigmoid output)
                model = NeuralNetScratch(
                    input_dim=input_dim,
                    hidden_dim1=hidden_dim1,
                    hidden_dim2=hidden_dim2,
                    output_dim=output_dim,
                    lr=args.lr,
                    epochs=args.epochs,
                    reg_lambda=0.001  # e.g., try 1e-3
                )

            elif args.model == 'perceptron':
                from perceptron import OneVsAllPerceptron
                model = OneVsAllPerceptron(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    lr=args.lr,
                    epochs=args.epochs,
                    reg_lambda=0.001
                )

            else:  # nn_pytorch
                import torch
                from nn_pytorch import NeuralNetPyTorch, train_pytorch_model
                # For multi-class => output_dim = 10, else = 1
                output_dim = num_classes if num_classes > 2 else 1
                model_pt = NeuralNetPyTorch(
                    input_dim=input_dim,
                    hidden_dim1=hidden_dim1,
                    hidden_dim2=hidden_dim2,
                    output_dim=output_dim
                ) #l2 regularization weight decay internal to class

            #############################
            # Train & time the model
            #############################
            start_time = time.time()

            if args.model == 'nn_scratch':
                # We must supply Y in correct format:
                # - multi-class => one-hot
                # - binary => shape (N,1)
                if num_classes > 2:
                    Y_sub = np.eye(num_classes)[y_sub]
                else:
                    Y_sub = y_sub.reshape(-1, 1)

                model.train(X_sub, Y_sub)

            elif args.model == 'perceptron':
                model.train(X_sub, y_sub)

            else:  # nn_pytorch
                #multi_class = (num_classes > 2)
                train_pytorch_model(
                    model_pt,
                    X_sub, y_sub,     # Train set
                    #None, None,       # No validation set
                    None, None,       # No test set
                    epochs= args.epochs,
                    lr= args.lr,
                    #batch_size=args.batch_size,  # batch_size
                    multi_class= (num_classes > 2),
                    weight_decay = 0.001
                )

            end_time = time.time()
            elapsed = end_time - start_time
            run_times.append(elapsed)

            #############################
            # Evaluate on test data
            #############################
            if args.model == 'nn_scratch':
                acc = evaluate_accuracy(model, X_test, y_test)
            elif args.model == 'perceptron':
                acc = evaluate_accuracy(model, X_test, y_test)
            else:
                # Evaluate PyTorch model
                model_pt.eval()
                import torch
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                y_test_t = torch.tensor(y_test)

                with torch.no_grad():
                    outputs = model_pt(X_test_t)
                    if num_classes > 2:
                        preds = torch.argmax(outputs, dim=1)
                        correct = (preds == y_test_t).sum().item()
                        acc = correct / len(y_test_t)
                    else:
                        probs = torch.sigmoid(outputs).squeeze()
                        preds = (probs > 0.5).long()
                        correct = (preds == y_test_t).sum().item()
                        acc = correct / len(y_test_t)

            run_accuracies.append(acc)

        #############################
        # Summarize fraction result
        #############################
        mean_time = np.mean(run_times)
        std_time = np.std(run_times)
        mean_acc = np.mean(run_accuracies)
        std_acc = np.std(run_accuracies)

        mean_err = 1.0 - mean_acc
        std_err = std_acc  # std of error = std of accuracy

        results.append({
            'fraction': frac,
            'mean_time': mean_time,
            'std_time': std_time,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'mean_err': mean_err,
            'std_err': std_err
        })

        print(f"\n=== Fraction: {frac:.1f} (N={N}) ===")
        print(f"Train Time: mean={mean_time:.3f}s, std={std_time:.3f}s")
        print(f"Test Accuracy: mean={mean_acc:.3f}, std={std_acc:.3f}")
        print(f"Test Error: mean={mean_err:.3f}, std={std_err:.3f}")

    print("\n===== Final Aggregated Results =====")
    for r in results:
        frac = r['fraction']
        print(f"\nFraction = {frac:.1f} => Using {int(len(X_train)*frac)} samples")
        print(f"  Time: mean={r['mean_time']:.3f}s, std={r['std_time']:.3f}s")
        print(f"  Accuracy: mean={r['mean_acc']:.3f}, std={r['std_acc']:.3f}")
        print(f"  Error: mean={r['mean_err']:.3f}, std={r['std_err']:.3f}")
    
    # =============================================================================
    # Visualization of final aggregated results
    # =============================================================================
    # Extract plotting lists from results
    fractions_list = [r['fraction'] for r in results]
    time_list = [r['mean_time'] for r in results]
    acc_list = [r['mean_acc'] for r in results]
    err_list = [r['mean_err'] for r in results]

    # 1) Plot Time vs Fraction
    plt.figure(figsize=(6, 4))
    plt.plot(fractions_list, time_list, marker='o')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Mean Training Time (s)')
    plt.title('Training Time vs. Fraction of Data')
    plt.grid(True)
    plt.show()

    # 2) Plot Accuracy vs Fraction
    plt.figure(figsize=(6, 4))
    plt.plot(fractions_list, acc_list, marker='o', color='g')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Mean Accuracy')
    plt.title('Accuracy vs. Fraction of Data')
    plt.grid(True)
    plt.show()

    # (3) Plot Error vs Fraction
    plt.figure(figsize=(6, 4))
    plt.plot(fractions_list, err_list, marker='o', color='r')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Mean Error')
    plt.title('Error vs. Fraction of Data')
    plt.grid(True)
    plt.show()

    # (4) Plot standard deviation of accuracy vs. fraction
    std_acc_list = [r['std_acc'] for r in results]

    plt.figure(figsize=(6, 4))
    plt.plot(fractions_list, std_acc_list, marker='o', color='m')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Std. Dev. of Accuracy')
    plt.title('Std. Dev. of Accuracy vs. Fraction of Data')
    plt.grid(True)
    plt.show()
    # =============================================================================


if __name__ == "__main__":
    main()
