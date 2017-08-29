import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  from IPython.core.debugger import Tracer
  #Tracer()()
  num_train, _ = X.shape
  _, num_class = W.shape
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    sum_scores = np.sum(exp_scores)
    prob = exp_scores/sum_scores
    loss += (-np.log(prob[y[i]]))
    for j in xrange(num_class):
      if j == y[i]:
        dW[:, y[i]] += -X[i] + prob[y[i]]*X[i]
      else:
        dW[:, j] += prob[j]*X[i]
        
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train # add by me 
  dW += 2*reg*W # add by me 
 
                   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  from IPython.core.debugger import Tracer
  #Tracer()()    
  num_train, _ = X.shape
  _, num_class = W.shape
  scores = X.dot(W)
  scores -= np.expand_dims(np.max(scores, axis=1), axis=1)
  exp_scores = np.exp(scores)
  sum_scores = np.sum(exp_scores, axis=1)
  prob = exp_scores/np.expand_dims(sum_scores, axis=1)
  loss += np.sum(-np.log(prob[np.arange(num_train), y]))
  #Tracer()()
  prob[np.arange(num_train), y] -= 1
  dW = X.T.dot(prob)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train # add by me 
  dW += 2*reg*W # add by me 
 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

