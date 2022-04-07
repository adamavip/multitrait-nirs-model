import numpy as np


def dataaugment(x, betashift=0.05, slopeshift=0.05, multishift=0.05):
    # Shift of baseline
    # calculate arrays
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    # Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    # Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x


def perform_dataaugmentation(X_train, y_train):
    shift = np.std(X_train)*0.1

    X_train_aug = np.repeat(X_train, repeats=100, axis=0)
    X_train_aug = dataaugment(X_train_aug, betashift = shift, slopeshift = 0.05, multishift = shift)

    y_train_aug = np.repeat(y_train, repeats=100, axis=0) # y_train is simply repeated

    return X_train_aug, y_train_aug
