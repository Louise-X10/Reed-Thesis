import pandas as pd
import numpy as np
import time

from ucimlrepo import fetch_ucirepo
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from scipy.special import softmax

# Set seed for random sampling
np.random.seed()

# fetch adult dataset, return train-test and X-y split


def load_adult():
    adult = fetch_ucirepo(id=2)
    # print(adult.metadata)
    # print(adult.variables)
    # print('='*40)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets
    adult = pd.concat([X, y], axis=1)

    # Drop all records with missing values
    adult.dropna(inplace=True)
    adult.reset_index(drop=True, inplace=True)
    # Drop fnlwgt, not interesting for ML
    adult.drop('fnlwgt', axis=1, inplace=True)
    adult.drop('education', axis=1, inplace=True)

    # Convert objects to categories
    obj_columns = adult.select_dtypes(['object']).columns
    adult[obj_columns] = adult[obj_columns].astype('category')

    num_columns = adult.select_dtypes(['int64']).columns
    adult[num_columns] = adult[num_columns].astype('float64')
    for c in num_columns:
        adult[c] /= (adult[c].max()-adult[c].min())

    adult['income'] = adult['income'].cat.codes
    # Correction: category 0, 1 is <=50k, categories 2, 3 is >50k
    adult['income'][adult['income'] == 1] = 0
    adult['income'][adult['income'] == 2] = 1
    adult['income'][adult['income'] == 3] = 1

    adult.replace(['Divorced',
                   'Married-AF-spouse',
                   'Married-civ-spouse',
                   'Married-spouse-absent',
                   'Never-married',
                   'Separated',
                   'Widowed'
                   ],
                  ['not married',
                   'married',
                   'married',
                   'married',
                   'not married',
                   'not married',
                   'not married'
                   ], inplace=True)

    cat_columns = adult.select_dtypes(['category']).columns
    adult = pd.get_dummies(adult, columns=cat_columns)
    X = np.array(adult.drop('income', axis=1))
    y = np.array(adult['income'])
    y = np.eye(2)[y]

    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    # Fix seed for train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=11)
    return X_train, X_test, y_train, y_test


# disable training messages
tf.keras.utils.disable_interactive_logging()

X_train, X_test, y_train, y_test = load_adult()

# Draw training data sample of size n from X_train, excluding row i
# @return: array of indices


def sample_dataset(i: int, n: int):
    sample_idx = np.random.randint(0, X_train.shape[0]-1, size=n)
    # if sampled idx >= i, then increment the idx, effectively excluding i from sampling
    sample_idx[sample_idx >= i] += 1
    return sample_idx

# Create neural network with 2 hidden layers, 1 output layer with 2 output nodes
def create_nn(regularizer=None):
    input_data = Input(shape=X_train[0].shape)
    x = Dense(40, activation='relu',
              kernel_regularizer=regularizer)(input_data)
    x = Dense(40, activation='relu', kernel_regularizer=regularizer)(x)
    output = Dense(2, kernel_regularizer=regularizer)(x)

    model = Model(input_data, output)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model

# Train model (without regularization) on given dataset
def train_model(X_train, y_train, X_test, y_test, epochs, batch_size):
    model = create_nn()
    # Train network until convergence
    start_time = time.time()
    r = model.fit(X_train,
                  y_train,
                  validation_data=(X_test, y_test),
                  epochs=epochs,
                  batch_size=batch_size
                  )
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("time elapsed from model training: ", time_elapsed)
    return model

# Train model (with regularization) on given dataset
def train_model_reg(X_train, y_train, X_test, y_test, epochs, batch_size):
    model = create_nn(regularizer='l2')
    # Train network until convergence
    start_time = time.time()
    r = model.fit(X_train,
                  y_train,
                  validation_data=(X_test, y_test),
                  epochs=epochs,
                  batch_size=batch_size
                  )
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    print("time elapsed from model training: ", time_elapsed)
    return model
