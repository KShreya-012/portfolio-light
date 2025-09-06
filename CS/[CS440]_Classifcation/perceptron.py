
# perceptron.py
import numpy as np

#Perceptron classifier (from scratch)

#### Usage (for digits, say num_classes=10; for face vs. non-face, num_classes=2)
class OneVsAllPerceptron:
    """
    CLASS "one-vs-all" perceptron for multi-class classification.

    Let us train one binary perceptron for each class:
        For class c, examples labeled c are treated as +1,
        all others as -1.

    W : np.ndarray of shape (num_classes, input_dim)
        W[c] are the weights for class c.
    b : np.ndarray of shape (num_classes,)
        b[c] is the bias for class c
    """

    def __init__(self, input_dim, num_classes, lr=0.01, epochs=10, use_bias=True, reg_lambda=0.0): #constructor
        """
        Parameters:
        -----------
        input_dim : int
            Number of features (e.g., pixels) per sample.
        num_classes : int
            The number of distinct classes.
        lr : float
            Learning rate.
        epochs : int
            Number of passes through the training set.
        use_bias : bool
            Whether to use a separate bias term for each class. If False, biases remain 0.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.use_bias = use_bias
        self.reg_lambda = reg_lambda      # ← store λ

        #initialize the weights as a 2D array of shape (num_classes, input_dim)
        #W[c] will represent the weight vector for class c of dim (input_dim, )
        self.W = np.zeros((num_classes, input_dim))

        #initialize bias
        if self.use_bias:
            self.b = np.zeros(num_classes) 
        else:
            self.b = None

    def train(self, X, y):
        """
        Train the one-vs-all perceptron.

        Parameters:
        -----------
        X : np.ndarray of shape (N, input_dim)
            Training data (N samples, each with input_dim features).
        y : np.ndarray of shape (N,)
            Labels for each sample, integer in [0..num_classes-1].
        """
        N = len(X) # num of train samples

        for epoch in range(self.epochs):
            # Shuffle the data each epoch to ensures we visit training samples in a random order for each epoch
            indices = np.arange(N) #(0,...,N-1)
            np.random.shuffle(indices)

            for i in indices:
                x_i = X[i] #feature vector for i-th sample
                label_i = y[i] #true class label for i-th sample

                # For each class c:
                for c in range(self.num_classes):
                    #L2 weight decay 
                    # Apply weight decay BEFORE computing the score
                    # W[c] ← (1 - lr * λ) · W[c]
                    if self.reg_lambda > 0.0:
                        self.W[c] *= (1 - self.lr * self.reg_lambda)

                    # binary target val: +1 if label == c, else -1
                    target = 1 if (label_i == c) else -1

                    # compute score = W[c].dot(x_i) + b[c]
                    score_c = np.dot(self.W[c], x_i)
                    if self.use_bias:
                        score_c += self.b[c]

                    # predicted sign (+1 or -1), "is ith sample in class c?" 
                    pred = 1 if score_c >= 0 else -1

                    # Perceptron update rule:
                    # if predicted_sign != true_sign (target) , update...
                    if pred != target:
                        # W[c] <- W[c] + lr * (target) * x_i
                        self.W[c] += self.lr * target * x_i

                        if self.use_bias:
                            # b[c] <- b[c] + lr * target
                            self.b[c] += self.lr * target

            # (debugging purposes) evaluate training accuracy per epoch
            # y_pred_train = self.predict(X)
            # train_acc = np.mean(y_pred_train == y)
            # print(f"Epoch {epoch+1}/{self.epochs}, Training Acc: {train_acc:.4f}")

    def predict(self, X):
        """
        Predict class labels for samples in X using the one-vs-all approach.

        Parameters:
        -----------
        X : np.ndarray of shape (M, input_dim)

        Returns:
        --------
        y_pred : np.ndarray of shape (M,)
            Predicted labels in [0..num_classes-1].
        """
        # Compute scores for each class: shape = (M, num_classes)
        scores = np.dot(X, self.W.T)  # shape (M, num_classes)
        if self.use_bias:
            scores += self.b  # adds bias per class (broadcast over M rows)

        # Choose the class with the highest score as the predicted label
        y_pred = np.argmax(scores, axis=1)
        return y_pred #array with predicted classes for each sample
