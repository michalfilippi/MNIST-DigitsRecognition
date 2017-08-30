
# coding: utf-8

# # Handwritten digits recognition using CNN

# This approach is inspired by article 'Multi-column Deep Neural Networks for
# Image Classification, Dan Cireşan, Ueli Meier, Juergen Schmidhuber, 2012'
# ([link](https://arxiv.org/abs/1202.2745)). The article describes currently
# (as in August 2017) the best performing model for handwritten digits
# recognition.
#  
# Since I don't have necessary hardware and/or time to train the model in
# original size I had to significantly scale things down. Therefore the results
# are not comparable to original paper. The approach presented here is just
# merely inspired by original work.

import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# Keras offers a simple way for importing MNIST dataset. Let's use it.
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

# Convolution layer which we will use expects every input sample to have three
# dimensions (width, height, depth).
# Therefore we need to transform x from (n, 28, 28) into (n, 28, 28, 1).

x_train = np.expand_dims(x_train_raw, axis=3)
x_test = np.expand_dims(x_test_raw, axis=3)

# Also simple preprocessing is handy. We'd like to have all inputs between 0
# and 1.
# 
# But our dataset has values in range (0, 255), so we should scale all values in
# all samples.

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# Targets in 'y_test' and 'y_train' are numbers 0..9, but we need our targets to
# match 10 output neurons which we will use in our neural network (every one of
# them corresponds to one output number).
# We will transform target 'k' to one-hot-vector with length of 10 and value 1
# on k-th position.

y_train = to_categorical(y_train_raw, 10)
y_test = to_categorical(y_test_raw, 10)


def create_model(verbose=False):
    """Creates predefined model and returns it.
    
    :param verbose: Prints model summary if verbose is True
    :return: Keras model object.
    """
    
    # define input and it's shape
    nn_input = Input(name='nn_input', shape=(28, 28, 1), dtype='float32')
    
    # first layer of convolution and pooling
    con_1 = Conv2D(filters=80, kernel_size=(5, 5), activation='tanh')(nn_input)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(con_1)
    
    # second layer of convolution and pooling
    con_2 = Conv2D(filters=80, kernel_size=(3, 3), activation='tanh')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(con_2)
    
    # flatten outputs from previous layers in order to pass them to dense layers
    flatten = Flatten()(pool_2)
    
    # fully connected layers 
    dense_1 = Dense(80, activation='tanh')(flatten)
    dense_2 = Dense(10, activation='softmax')(dense_1)
    
    # model construction and compilation
    model = Model(inputs=nn_input, outputs=dense_2)
    model.compile(loss=categorical_crossentropy, 
                  optimizer=Adam(), 
                  metrics=['accuracy'])
    
    # print model architecture
    if verbose:
        model.summary()
    
    return model


def test_model(model, verbose=False):
    """ Tests model passed in parameter 'model' on test data saved in variables 
    'x_test' and 'y_test'. 
    
    :param model: Keras model to be tested.
    :param verbose: Prints error rate if verbose is True.
    :return: tuple (e, m) where e is measured error rate and m is a list of all 
    misclassified samples by given model. 
    """

    predictions = model.predict(x_test, verbose=0)
    
    misclassified = []
    for x, y, p in zip(x_test_raw, y_test_raw, predictions):
        predicted_class = np.argmax(p)
        if predicted_class != y:
            # miss, remember missed sample
            misclassified.append((x, y, predicted_class))
    
    # calculate error rate
    error_rate = len(misclassified) / len(x_test) * 100
    
    if verbose:
        print('Error rate: {}%'.format(error_rate))
    
    return error_rate, misclassified


# First we create simple wrapper for the ensemble model.
class EnsembleModel:
    """Simple wrapper over Keras Model objects to provide simple API for 
    testing"""

    def __init__(self, models, combining_method='voting'):
        self.models = models

        if combining_method == 'voting':
            self.combinig_method = EnsembleModel.voting
        elif combining_method == 'averaging':
            self.combinig_method = EnsembleModel.averaging
        else:
            print('Unknown combining method:', combining_method)
            print('Fallback to voting.')
            self.combinig_method = EnsembleModel.voting

    def predict(self, x, batch_size=32, verbose=0):
        predictions = [m.predict(x, batch_size, verbose) for m in self.models]
        
        return self.combinig_method(predictions)

    @staticmethod
    def averaging(predictions):
        # sum all predictions
        final_prediction = np.zeros_like(predictions[0])
        for p in predictions:
            final_prediction = np.add(p, final_prediction)
        
        # normalize
        final_prediction /= final_prediction.sum(axis=1, keepdims=1)

        return final_prediction

    @staticmethod
    def voting(predictions):
        # sum all votes
        final_prediction = np.zeros_like(predictions[0])
        for p in predictions:
            one_hot = EnsembleModel._prediction_to_one_hot(p)
            final_prediction = np.add(one_hot, final_prediction)

        # normalize
        final_prediction /= final_prediction.sum(axis=1, keepdims=1)

        return final_prediction

    @staticmethod
    def _prediction_to_one_hot(prediction):
        one_hot = np.zeros_like(prediction)
        one_hot[np.arange(len(prediction)), prediction.argmax(1)] = 1
        
        return one_hot


# We will independently train 5 models on train data
n_models = 5
models = []
for i in range(n_models):
    m = create_model()
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)
    m.fit(x_train, y_train, batch_size=256, epochs=50,
          validation_split=0.05, callbacks=[es], verbose=0)
    models.append(m)

# Create ensemble model with averaging scheme and test it's performance on test
# data

em_a = EnsembleModel(models, 'averaging')
result_em_a = test_model(em_a, True)
