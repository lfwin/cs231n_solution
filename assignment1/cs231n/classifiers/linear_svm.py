import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  #from IPython.core.debugger import Tracer
  
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #Tracer()()
        dW[:, j] += X[i] # add by me 
        dW[:, y[i]] -= X[i] # add by me 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  dW /= num_train # add by me 
  dW += 2*reg*W # add by me 


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  from IPython.core.debugger import Tracer
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train), y]
  diff = scores - np.reshape(correct_class_score, (num_train, 1))
  losses = np.maximum(0, diff + 1)
 
  loss = (np.sum(losses)-num_train)/num_train + reg * np.sum(W * W) # - num_train
  # due to sum more losses when y[i]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #Tracer()()
  L = diff + 1
  L[L<0] = 0
  L[L>0] = 1
  #filter = (diff + 1) > 0 # for grad
  #Tracer()()
  L[np.arange(0, scores.shape[0]), y] = 0
  L[np.arange(0, scores.shape[0]), y] = -1 * np.sum(L, axis=1)
  #Tracer()()
  dW = np.dot(X.T, L)
  dW /= num_train # add by me 
  dW += 2*reg*W # add by me 
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
