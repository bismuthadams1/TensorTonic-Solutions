import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X, W, b = np.asarray(X), np.asarray(W), np.asarray(b)

    forward = X @ W + b

    return forward.tolist()
        