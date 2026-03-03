import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    X = np.asarray(X) # B,D 4,1
    y = np.asarray(y) # B
    
    b = 0.0
    w = np.zeros(X.shape[1]) #.reshape(1,-1) # 1,D (1,1)

    steps_count = 0

    while steps_count < steps:
        p = _sigmoid( X @ w + b) # X(B,D)[4,1] * w(1,D)[1,1] ==  B,
        dLdw = 1/(X.shape[0])*np.transpose(X) @ (p-y)# D,B[1,4] * B[4]
        w = w - lr * dLdw # [1,1]

        # w.reshape(1,-1)

        dLdb = 1/(X.shape[0])*(p-y).sum()
        b = b - lr * dLdb

        steps_count += 1
    

    return w, b
            
        