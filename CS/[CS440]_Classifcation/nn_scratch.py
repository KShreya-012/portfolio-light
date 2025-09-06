
#nn_scratch.py
import numpy as np

#Three-layer neural network (from scratch, with manual backprop)

# Activation fns (experiment)
#----------------------------
def relu(z):
    return np.maximum(0, z)

def d_relu(a):
    #Derivative of ReLU wrt z. Here a = relu(z).
    return (a > 0).astype(float) #deriv is 1 if z > 0, else 0. 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(a):
    #derivative of sigmoid wrt z, given a = sigmoid(z)
    return a * (1 - a)

def softmax(z): #z: (N, num_classes)
    
    # For numerical stability, subtract max over each row
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) #pr dist for each row (sample)
        #returns: (N, num_classes)


# Neural Network class
#----------------------------
class NeuralNetScratch:
    """
    3-layer NN: Input -> Hidden1 -> Hidden2 -> Output
    reLU activation fn used on both hidden layers
    Softmax or Sigmoid output based on classification task (Multi-class or Binary)
    Includes L2 regularization (weight decay).
    """

    def __init__(self, 
                 input_dim, #number of input features
                 hidden_dim1, #neurons in hidden layer 1
                 hidden_dim2, #neurons in hidden layer 2
                 output_dim, #if >1 => multi-class (softmax), if =1 => binary (sigmoid)
                 lr=0.01, #learning rate
                 epochs=10, # num of train epochs
                 batch_size=32, #mini-batch size
                 reg_lambda=0.0): #L2 regularization strength (lambda). [edge case: lambda=0 => no regularization]

        self.input_dim = input_dim
        self.h1 = hidden_dim1
        self.h2 = hidden_dim2
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda

        # Multi-class or binary (digits or face task)?
        self.is_multiclass = (self.output_dim > 1)

        # "He init" (fancy term) := 
            #It initializes weights from a normal distribution with a mean of 0 and a variance of 2/n, where 'n' is the number of input units to a neuron. 
            #The variance is inversely proportional to the number of input units, ensuring that the gradients don't become too small or too large during training
        #weights W are initialized as: W ~ N(0, sqrt(2 / n))
        limit1 = np.sqrt(2.0 / self.input_dim)
        self.W1 = np.random.randn(self.input_dim, self.h1) * limit1 #weight matrix
    
        self.b1 = np.zeros((1, self.h1)) #initialize bias to 0 vec

        limit2 = np.sqrt(2.0 / self.h1)
        self.W2 = np.random.randn(self.h1, self.h2) * limit2
        self.b2 = np.zeros((1, self.h2))

        limit3 = np.sqrt(2.0 / self.h2)
        self.W3 = np.random.randn(self.h2, self.output_dim) * limit3
        self.b3 = np.zeros((1, self.output_dim))

    #FORWARD PASS! 
    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1 #pre-activation for the first hidden layer
        A1 = relu(Z1) #activation for the first hidden layer

        Z2 = np.dot(A1, self.W2) + self.b2 #pre-activation for second hidden layer
        A2 = relu(Z2) #activation for second hidden layer

        Z3 = np.dot(A2, self.W3) + self.b3 #pre-activation for the output layer
        if self.is_multiclass:
            A3 = softmax(Z3)
        else:
            A3 = sigmoid(Z3)

        return Z1, A1, Z2, A2, Z3, A3 #for use in backprop

    def compute_loss(self, A3, Y):
        """
        Cross-entropy + L2 regularization
        A3: (N, output_dim)
        Y:  if multi-class => shape (N, output_dim), else (N, 1)
        """
        eps = 1e-12 #avoid log(0)
        N = len(Y) #number of samples in the batch or dataset

        # Main cross-entropy
        if self.is_multiclass:
            # multi-class cross-entropy: -sum(Y*log(A3))/N
            loss_ce = -np.sum(Y * np.log(A3 + eps)) / N
        else:
            # binary cross-entropy: -sum(Y*log(A3)+(1-Y)*log(1-A3))/N
            loss_ce = -np.sum(Y * np.log(A3 + eps) + (1 - Y)*np.log(1 - A3 + eps)) / N

        # L2 regularization term: reg_lambda/(2*N) * (||W1||^2 + ||W2||^2 + ||W3||^2)
        #only regularize weights W1,W2,W3 not biases.
        loss_l2 = (self.reg_lambda / (2*N)) * (
            np.sum(np.square(self.W1)) + 
            np.sum(np.square(self.W2)) + 
            np.sum(np.square(self.W3))
        )

        return loss_ce + loss_l2 #total loss = cross-entropy + L2 regularization

    #BACKPROPEGATION & L2
    def backward(self, X, Y, Z1, A1, Z2, A2, Z3, A3):
        #use the stored forward-pass vals to compute gradients and update weights
        N = len(X) # batch size

        # dZ3 Output Layer Error
        if self.is_multiclass:
            dZ3 = (A3 - Y)  # (N, output_dim)
        else:
            dZ3 = (A3 - Y)  # (N,1)

        #gradient for W3 and b3
        dW3 = np.dot(A2.T, dZ3) / N
        db3 = np.sum(dZ3, axis=0, keepdims=True) / N

        # L2 gradient term for W3
        dW3 += (self.reg_lambda / N) * self.W3 #add L2 penalty derivative

        #backpropagating to layer 2
        dA2 = np.dot(dZ3, self.W3.T)   # (N, h2)
        dZ2 = dA2 * d_relu(A2) #chain rule
        #gradient for W2 and b2
        dW2 = np.dot(A1.T, dZ2) / N
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N

        # L2 for W2
        dW2 += (self.reg_lambda / N) * self.W2

        #backpropegate to layer 1
        dA1 = np.dot(dZ2, self.W2.T)  # (N, h1)
        dZ1 = dA1 * d_relu(A1)
        #dradient for W1, b1
        dW1 = np.dot(X.T, dZ1) / N
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N

        # L2 for W1
        dW1 += (self.reg_lambda / N) * self.W1

        # Gradient updates
            #new_W = (old_W - lr) x dW
            #new_b = (old_b - lr) x db
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    #TRAIN! 
    def train(self, X, Y): 
        N = len(X)
        for epoch in range(self.epochs):
            # Shuffle each epoch
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            # split data into mini-batches of size "batch_size" 
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                Z1, A1, Z2, A2, Z3, A3 = self.forward(X_batch) #forward pass on the mini-batch to recover all intermediate outputs
                _ = self.compute_loss(A3, Y_batch)  # optionally store/track
                self.backward(X_batch, Y_batch, Z1, A1, Z2, A2, Z3, A3) #backward pass to compute gradients and update weights

            # (for debugging purposes) print progress
            if (epoch + 1) % 10 == 0:
                Z1_full, A1_full, Z2_full, A2_full, Z3_full, A3_full = self.forward(X)
                loss_full = self.compute_loss(A3_full, Y)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss (train): {loss_full:.4f}")

    def predict(self, X):
        _, _, _, _, _, A3 = self.forward(X) # forward pass
        if self.is_multiclass:
            return np.argmax(A3, axis=1) #picks the class with the highest probability
        else:
            return (A3 > 0.5).astype(int).flatten() #predicts 1 if the output > 0.5, else 0
