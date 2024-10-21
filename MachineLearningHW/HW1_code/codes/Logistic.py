import numpy as np

class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
    def gen_features(self, X): # part d of number 4
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        X_out = np.hstack((np.ones((N,1)), X))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out  
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) # (d + 1) x 1, all the features
        N,d = X.shape 
        
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        X_new = self.gen_features(X)
        for row in range(N):
            x = X_new[row]
            h = self.w.T @ x # X_new[index] is (d + 1) x 1 and w.T is 1 x (d + 1)
            loss += np.log(1 + np.exp(h))
            val = 0
            if (y[row] == 1):
                loss -= h
                val = x.reshape(grad.shape)
            
            #sigmoid = 1/(1 + np.exp(-h)) # scalar
            #grad = grad + ((sigmoid - (y[row])) * x).reshape(grad.shape) # (d + 1) x 1
            grad += (((np.exp(h)) / (1 + np.exp(h))) * x).reshape(grad.shape) - val

        loss /= N
        grad /= N
        # Calculate the gradient and save it as grad
        # ================================================================ #
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                # ================================================================ #
                # YOUR CODE HERE:
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                # ================================================================ #
                randomIndices = np.random.choice(N, size = batch_size, replace = False)
                X_batch = X[randomIndices]
                y_batch = y[randomIndices]
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss = 0.0
                grad = np.zeros_like(self.w)
                # ================================================================ #
                # YOUR CODE HERE: 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                # ================================================================ #
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w = self.w - (eta * grad)
                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #
                loss_history.append(loss)
        return loss_history, self.w
    
    def predict(self, X): # 4 part e
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0]) # N x 1 # was previously (X.shape[0] + 1)
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X_new = self.gen_features(X) # X_new is N x (d + 1)
        # use the sigmoid function
        for index in range(len(y_pred)):
            a = self.w.T @ X_new[index] # X_new[index] is (d + 1) x 1 and w.T is 1 x (d + 1)
            sigmoid = 1/(1 + np.exp(-a))
            if (sigmoid > 0.5): # positive, classify as +1
                 y_pred[index] = 1
            elif (sigmoid < 0.5): # negative, classify as -1
                 y_pred[index] = -1
            else: # = 0, undecidable
                 y_pred[index] = 0
        y_pred = y_pred.reshape((len(y_pred), 1))
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred