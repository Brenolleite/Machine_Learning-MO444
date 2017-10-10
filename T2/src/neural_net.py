import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, train_test_split, learning_curve
import keras
from keras.models import Sequential
from keras.layers import Dense
import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
import gradient_checking as grad

def create_model(hidden_layers, n_neurons_input, n_neurons, activation, final_activation, loss, optimizer, batch_size, epochs, generate_confusionMatrix):
    # Create neural network (input: 3072)
    model = Sequential()

    model.add(Dense(n_neurons, activation=activation, input_shape=(n_neurons_input,)))
    model.add(Dense(n_neurons, activation=activation))
    for i in range(hidden_layers):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(10, activation=final_activation))

    #model.summary()

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def predict(model, data, labels, verbose):
    class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Predict data on the validation
    predictions = model.predict(data, verbose=2)
    pred = np.argmax(predictions, axis=1)
    if verbose:
        cm = confusion_matrix(np.argmax(labels,axis=1), pred)
        metrics.plot_confusion_matrix(cm, classes=class_name, title='Confusion Matrix')
        metrics.plot_confusion_matrix(cm, classes=class_name, normalize=True, title='Normalized Confusion matrix')

    # Compute metrics
    score = model.evaluate(data, labels, verbose=0)
    metrics.print_acc_nn(score)

    return score[1]


def kfold(model_params, train_data, train_labels, n_folds, verbose, grad_check):
    # Create array for storage models and errors
    models = []
    iterations = 1
    batch_size = model_params[7]
    epochs = model_params[8]
    generate_confusionMatrix = model_params[9]

    # Create KFold validation
    kf = KFold(n_splits = n_folds)
    fold = 0

    for train, validate in kf.split(train_data, train_labels):

        # Create the model using the params
        model = create_model(*model_params)

        if verbose:
            print("================== Fold {0} ==========================".format(fold))
        fold += 1

        for i in range(iterations):
            # Train our model using train set
            model.fit(train_data[train], train_labels[train],
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(train_data[validate], train_labels[validate]))

            # Verify on validation set
            acc = predict(model, train_data[validate], train_labels[validate], generate_confusionMatrix)

            # Check gradient
            if grad_check:
                grad.check(model, train_data)

        # Store model and erros related to it
        models.append([model, acc])

    # Print avr error
    print("Average accuracy on {0}-Fold \n============================\n".format(n_folds))
    models = np.array(models)
    accs = np.sum(models[:, 1], 0)/n_folds
    metrics.print_acc(accs)

    return models


def test(model_params, train_data, train_labels, test_data, test_labels):
    models = []
    iterations = 1
    batch_size = model_params[7]
    epochs = model_params[8]
    generate_confusionMatrix = model_params[9]

    model = create_model(*model_params)
    model.fit(train_data, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data, test_labels))

    acc = predict(model, test_data, test_labels, generate_confusionMatrix)


def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)