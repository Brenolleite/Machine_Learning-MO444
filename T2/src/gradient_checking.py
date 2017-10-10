import keras
import keras.backend as K
import numpy as np
from scipy.optimize import check_grad

# For checking porposes the net needs to be
# configured with 1 hidden layer only

# Create softmax function used to grad check by scipy
def softmax(self, Z):
        Z = np.maximum(Z, -1e3)
        Z = np.minimum(Z, 1e3)
        numerator = np.exp(Z)
        return numerator / sum(numerator, axis=1).reshape((-1,1))

def check(model, input):
    # Get variables from model
    variables = model.trainable_weights

    # Get loss function at last layer
    loss = model.output

    # Get gradients using loss function and variables
    gradients = K.gradients(loss, variables)

    # Expecify tensorFlow session to get tensor values
    sess = K.get_session()

    # Use the last layer to check the gradient
    check_grad(softmax, gradients[6].eval(session=sess), input)