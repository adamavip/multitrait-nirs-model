import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, Concatenate, concatenate, Input
from tensorflow.keras.layers import Reshape, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Input, LocallyConnected1D, Activation
from tensorflow.keras.layers import LSTM, GRU, TimeDistributed, GaussianDropout, AlphaDropout, GaussianNoise
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax, Adadelta
from tensorflow.keras.utils import plot_model
from keras.regularizers import l1,l2
import numpy as np
import os
from datetime import datetime
import pickle

def denseModel(feature_count, x_dim):
    model = Sequential()
    model.add(Dropout(0.02, input_shape=(feature_count, x_dim)))
    model.add(GaussianDropout(0.3, input_shape=(feature_count, x_dim)))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Flatten())
    return model


def convModel(feature_count, x_dim):
    model = Sequential()
    model.add(Dropout(0.002, input_shape=(feature_count, x_dim)))
    # model.add(GaussianDropout(0.2))
    # model.add(GaussianNoise(0.2))
    model.add(Conv1D (filters=32, kernel_size=5, strides=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(0.02))

    model.add(Conv1D (filters=64, kernel_size=5, strides=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout1D(0.02))

    model.add(Conv1D (filters=128, kernel_size=5, strides=1, activation='relu'))
    model.add(SpatialDropout1D(0.02))
    model.add(BatchNormalization())

    model.add(Conv1D (filters=256, kernel_size=5, strides=1, activation='relu'))
    model.add(SpatialDropout1D(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=l1(1e-7), bias_regularizer=l1(1e-7), activation='relu'))


    model.add(Dense(1,activation='linear'))
    return model


def convModel2(feature_count,x_dim):
    model = Sequential()
    #Adding a bit of GaussianNoise also works as regularization
    model.add(GaussianNoise(0.05, input_shape=(feature_count,x_dim)))
    #model.add(Dropout(0.002, input_shape=(feature_count, x_dim)))
    model.add(Conv1D(filters=24, kernel_size=15, use_bias=False, strides=2, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2,strides=1))
    model.add(SpatialDropout1D(0.02))

    model.add(Conv1D(filters=48,kernel_size=15 , use_bias=False, strides=2, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(SpatialDropout1D(0.02))

    model.add(Conv1D(filters=96,kernel_size=15, use_bias=False, strides=2, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2,strides=1))
    model.add(SpatialDropout1D(0.02))


    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, kernel_regularizer=l1(1e-7), bias_regularizer=l1(1e-7)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))


    model.add(Dense(128, kernel_regularizer=l1(1e-7), bias_regularizer=l1(1e-7)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.02))

    model.add(Dense(1,activation='linear'))
    return model

def convModel3(feature_count,x_dim):
    model = Sequential()
    model.add(Dropout(0.002, input_shape=(feature_count, x_dim)))
    model.add(Conv1D(filters=16, kernel_size=5, use_bias=False, strides=2, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2,strides=1))
    model.add(SpatialDropout1D(0.02))

    model.add(Conv1D(filters=36,kernel_size=15 , use_bias=False, strides=2, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(SpatialDropout1D(0.02))

    model.add(Conv1D(filters=56,kernel_size=20, use_bias=False, strides=2, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2,strides=1))
    model.add(SpatialDropout1D(0.02))


    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l1(1e-7), bias_regularizer=l1(1e-8)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))



    model.add(Dense(128, kernel_regularizer=l1(1e-7), bias_regularizer=l1(1e-8)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(1,activation='sigmoid'))
    return model

#Hyperparameters for the network
DENSE = 128
DROPOUT = 0.5
C1_K  = 8 #Number of kernels/feature extractors for first layer
C1_S  = 32 #Width of the convolutional mini networks
C2_K  = 16
C2_S  = 32
activation='relu'

#The model
def convModel5(feature_count,x_dim):
    model = Sequential()
    #Adding a bit of GaussianNoise also works as regularization
    model.add(GaussianNoise(0.05, input_shape=(feature_count,)))
    #First two is number of filter + kernel size
    model.add(Reshape((feature_count,x_dim) ))
    model.add(Conv1D(C1_K, (C1_S), activation=activation, padding="same"))
    model.add(Conv1D(C2_K, (C2_S), padding="same", activation=activation))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE, activation=activation))
    model.add(Dense(1, activation='linear'))

    return model

def define_convModel(cnnModel, X_train):
    NIRS_feature_count = X_train.shape[1]
    NIRS_dim = X_train.shape[2]

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy(["GPU:0"])
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        model = cnnModel(NIRS_feature_count, NIRS_dim)
        print("-Number of parameters: ", model.count_params())
        #Save model architecture
        #plot_model(training_model, to_file='./results/model_architecture.png', show_shapes=True, show_layer_names=True)
        #Compile model
        print('Model compilation\n')
        opt = Adam(learning_rate=3e-4)
        model.compile(loss='mean_squared_error', metrics=['mse'], optimizer=opt)
        #model.compile(loss='mse', optimizer=Adadelta(lr=0.01))

    return model

def train_model(model,X_train, y_train, X_val, y_val, model_type=None):

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=0, min_delta=1e-6, mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')

    if model_type=="ML":
        X_train = np.squeeze(X_train)
        y_train = np.squeeze(y_train)
        model.fit(X_train, y_train)

    elif model_type=="DL":

        print('Model training started...')
        history = model.fit(X_train, y_train,
                            epochs=500,
                            batch_size=16,
                            validation_data=(X_val,y_val),
                            verbose=0,
                            callbacks=[reduce_lr_loss,earlyStopping])
        print('Model training complete!')

    return model 


def save_model(model, model_type,target=None,sensor=None,crop=None):
    print("Start Saving Model...")
    now = datetime.now()
    time = now.strftime("%Y-%m-%d")
    #fn = crop + target + model_type + sensor + time
    fn = '_'.join([crop, target, model_type, sensor, time])
    save_dir = "../results/trained_models"

    if model_type=='ML':
        filename = fn + ".pkl"
        filename = os.path.join(save_dir,filename)
        # Open a file where you want to store the data
        file = open(filename, "wb")

        # Dump information into the file
        pickle.dump(model, file)

    elif model_type == 'DL':
        filename = os.path.join(save_dir,fn)
        model.save(filename,save_format='h5')

    print("Saving Complete!")
