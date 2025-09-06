
# nn_pytorch.py
import torch #main pytorch library
import torch.nn as nn # nn building blocks
import torch.nn.functional as F
import torch.optim as optim #optimization algos
# Define three-layer neural network (two hidden layers) w/ ReLU activation.

class NeuralNetPyTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(NeuralNetPyTorch, self).__init__()
        # defining the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Apply the activation functions explicitly in forward
        x = self.fc1(x)
        x = F.relu(x) # ReLU after fc1 
        #nn.Sigmoid() -> low accuracy on both datasets; opt for "rectified linear unit" ReLU R(z) = max(0,z). 
        x = self.fc2(x)
        x = F.relu(x) # ReLU after fc2
        x = self.fc3(x) # Output layer (no activation here by default)
        return x

"""
class NeuralNetPyTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(NeuralNetPyTorch, self).__init__()
        self.net = nn.Sequential( #nn.Sequential -> chain multiple layers and activation functions in a seq.
            #defining the layers
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(), #nn.Sigmoid() -> low accuracy on both datasets; opt for "rectified linear unit" ReLU R(z) = max(0,z). 
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(), 
            nn.Linear(hidden_dim2, output_dim)
            # For multi-class digits, no need to add nn.Softmax(dim=1) at the end, since we are using CrossEntropyLoss which applies log-softmax internally.
        )

    def forward(self, x):
        return self.net(x) #output layer
    ##### TD: modify class def so that activitation fns applied in forward not in definition
"""

def train_pytorch_model(
    model, #instance of NeuralNetPyTorch
    X_train, y_train, #Train data (features and labels)
    X_test, y_test,#Test data (features and labels)
    epochs=10, # number of training epochs
    lr=0.01, #learning  rate for optimizer
    batch_size=32, #for mini batch training
    multi_class=False, #boolean to identify task as binary or multi-class classification task
    weight_decay= 0.001  # param for L2 regularization strength
    #in train.py we set weight_decay = 0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Detects if GPU (“cuda”) is available, otherwise uses CPU.
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long if multi_class else torch.float32).to(device)
    
    #OPTIMIZER!
    #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Pass weight_decay=weight_decay to apply L2 regularization
        # If weight_decay > 0, that sets the L2 penalty term on the model's weights.
    
    #choose loss fn:
    if multi_class:
        #For multi-class classification (DIGITS)
        criterion = nn.CrossEntropyLoss()
    else:
        # For binary classification (FACES)
        criterion = nn.BCEWithLogitsLoss()  # use bc final layer has no sigmoid
        #if we had a sigmoid final layer then use nn.BCELoss()

    #Train time :) 
    for epoch in range(epochs):
        model.train() #train mode
        permutation = torch.randperm(X_train_t.size(0))
            #creates a random permutation of the training examples’ indices to shuffle the data for each epoch
        
        #mini-batch iterations
        for i in range(0, X_train_t.size(0), batch_size): #range(start,stop,step)
            #for every batch
            batch_idx = permutation[i:i+batch_size]
            batch_x = X_train_t[batch_idx] #data X
            batch_y = y_train_t[batch_idx] #label Y 
            batch_y.to(device)

            optimizer.zero_grad() #Clear gradients for each batch before backpropegating
            outputs = model(batch_x) #forward pass... model makes a prediction on data X

            #Fix shape mismatch if needed... 
            if outputs.dim() == 2 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)  
            batch_y = batch_y.view(-1) 

            loss = criterion(outputs, batch_y) #compute loss = loss_fn (predicted_label , true_label) 
            loss.backward() #backpropegate the loss we just computed; (computes gradients wrt/ each parameter automatically)
            optimizer.step() #update weights according to the optimizer’s rule (e.g. Adam or SGD)
            #print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


        # Evaluation (within each epoch): measure training accuracy
        model.eval() #deactivate training-specific behaviors

        with torch.no_grad(): #wrap loop with torch.no_grad() to save memory and speed up computation
            train_outputs = model(X_train_t) #forward pass
            if multi_class:
                pred_labels = torch.argmax(train_outputs, dim=1) #get pred class
                train_acc = (pred_labels == y_train_t).float().mean().item() #compare predictions to y_train_t to get accuracy
            else:
                pred_labels = torch.sigmoid(train_outputs).squeeze() > 0.5
                train_acc = (pred_labels.float() == y_train_t).float().mean().item()
                #pass through torch.sigmoid() to convert raw vals into pr. \in [0,1], then threshold at 0.5 to get pred classes

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")

    # Evaluate on X_test
    if X_test is not None and y_test is not None:
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.long if multi_class else torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            if multi_class: #choose predicted class using argmax
                pred_labels = torch.argmax(test_outputs, dim=1)
                test_acc = (pred_labels == y_test_t).float().mean().item()
            else: #apply sigmoid and threshold
                pred_labels = torch.sigmoid(test_outputs).squeeze() > 0.5
                test_acc = (pred_labels.float() == y_test_t).float().mean().item()
        print(f"Test Accuracy: {test_acc:.4f}")

    return model #finished training and weights have converged

#NOTES:

"""
For binary classification, set multi_class=False, so we use BCEWithLogitsLoss (if we had used sigmoid then BCELoss)

For multi-class digits, set multi_class=True ; multi-class classification task ==> rely on CrossEntropyLoss.
"""