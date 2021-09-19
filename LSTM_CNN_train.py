import sys
import cv2
import os
import numpy as np
from numpy.random import randint
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        ### Bildirim Seviyesi

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LSTM, Bidirectional

from tensorflow.keras.applications.vgg16 import VGG16

import warnings


print("tf version: ", tf.__version__)
print("np version: ", np.__version__)
print("np version: ", np.__file__)
print("OpenCv version: ", cv2.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('physical_devices: ', physical_devices[0])
tf.config.experimental.set_memory_growth(physical_devices[0], True)
warnings.simplefilter("ignore")

np.random.seed(0)

lr_epoch = 1000
opt = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt_d = Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
    if programPause == 'q':
        sys.exit()
        

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    

def load_dataset():
    # load dataset
    (trainX, trainy), (testX, testy) = cifar10.load_data()
    
    trainY = to_categorical(trainy)
    testY = to_categorical(testy)
    
    trainX = (trainX - 127.5) / 127.5
    testX = (testX - 127.5) / 127.5
    
    trainy_0 = trainy[np.array(np.where(trainy == 0))[0]]
    trainy_1 = trainy[np.array(np.where(trainy == 1))[0]]
    trainy_2 = trainy[np.array(np.where(trainy == 2))[0]]
    trainy_3 = trainy[np.array(np.where(trainy == 3))[0]]
    trainy_4 = trainy[np.array(np.where(trainy == 4))[0]]
    trainy_5 = trainy[np.array(np.where(trainy == 5))[0]]
    trainy_6 = trainy[np.array(np.where(trainy == 6))[0]]
    trainy_7 = trainy[np.array(np.where(trainy == 7))[0]]
    trainy_8 = trainy[np.array(np.where(trainy == 8))[0]]
    trainy_9 = trainy[np.array(np.where(trainy == 9))[0]]
    
    trainX_0 = trainX[np.array(np.where(trainy == 0))[0]]
    trainX_1 = trainX[np.array(np.where(trainy == 1))[0]]
    trainX_2 = trainX[np.array(np.where(trainy == 2))[0]]
    trainX_3 = trainX[np.array(np.where(trainy == 3))[0]]
    trainX_4 = trainX[np.array(np.where(trainy == 4))[0]]
    trainX_5 = trainX[np.array(np.where(trainy == 5))[0]]
    trainX_6 = trainX[np.array(np.where(trainy == 6))[0]]
    trainX_7 = trainX[np.array(np.where(trainy == 7))[0]]
    trainX_8 = trainX[np.array(np.where(trainy == 8))[0]]
    trainX_9 = trainX[np.array(np.where(trainy == 9))[0]]
    
    rd_choice_0 = np.random.choice(5000, 1000, replace=False)
    rd_choice_1 = np.random.choice(5000, 1000, replace=False)
    rd_choice_2 = np.random.choice(5000, 1000, replace=False)
    rd_choice_3 = np.random.choice(5000, 1000, replace=False)
    rd_choice_4 = np.random.choice(5000, 1000, replace=False)
    rd_choice_5 = np.random.choice(5000, 1000, replace=False)
    rd_choice_6 = np.random.choice(5000, 1000, replace=False)
    rd_choice_7 = np.random.choice(5000, 1000, replace=False)
    rd_choice_8 = np.random.choice(5000, 1000, replace=False)
    rd_choice_9 = np.random.choice(5000, 1000, replace=False)
    
    valX_0 = trainX_0[rd_choice_0]
    valX_1 = trainX_1[rd_choice_1]
    valX_2 = trainX_2[rd_choice_2]
    valX_3 = trainX_3[rd_choice_3]
    valX_4 = trainX_4[rd_choice_4]
    valX_5 = trainX_5[rd_choice_5]
    valX_6 = trainX_6[rd_choice_6]
    valX_7 = trainX_7[rd_choice_7]
    valX_8 = trainX_8[rd_choice_8]
    valX_9 = trainX_9[rd_choice_9]
    
    valy_0 = trainy_0[rd_choice_0]
    valy_1 = trainy_1[rd_choice_1]
    valy_2 = trainy_2[rd_choice_2]
    valy_3 = trainy_3[rd_choice_3]
    valy_4 = trainy_4[rd_choice_4]
    valy_5 = trainy_5[rd_choice_5]
    valy_6 = trainy_6[rd_choice_6]
    valy_7 = trainy_7[rd_choice_7]
    valy_8 = trainy_8[rd_choice_8]
    valy_9 = trainy_9[rd_choice_9]
    
    train_choice_0 = np.setdiff1d(np.arange(5000), rd_choice_0)
    train_choice_1 = np.setdiff1d(np.arange(5000), rd_choice_1)
    train_choice_2 = np.setdiff1d(np.arange(5000), rd_choice_2)
    train_choice_3 = np.setdiff1d(np.arange(5000), rd_choice_3)
    train_choice_4 = np.setdiff1d(np.arange(5000), rd_choice_4)
    train_choice_5 = np.setdiff1d(np.arange(5000), rd_choice_5)
    train_choice_6 = np.setdiff1d(np.arange(5000), rd_choice_6)
    train_choice_7 = np.setdiff1d(np.arange(5000), rd_choice_7)
    train_choice_8 = np.setdiff1d(np.arange(5000), rd_choice_8)
    train_choice_9 = np.setdiff1d(np.arange(5000), rd_choice_9)
    
    trainX_0 = trainX_0[train_choice_0]
    trainX_1 = trainX_1[train_choice_1]
    trainX_2 = trainX_2[train_choice_2]
    trainX_3 = trainX_3[train_choice_3]
    trainX_4 = trainX_4[train_choice_4]
    trainX_5 = trainX_5[train_choice_5]
    trainX_6 = trainX_6[train_choice_6]
    trainX_7 = trainX_7[train_choice_7]
    trainX_8 = trainX_8[train_choice_8]
    trainX_9 = trainX_9[train_choice_9]
    
    trainy_0 = trainy_0[train_choice_0]
    trainy_1 = trainy_1[train_choice_1]
    trainy_2 = trainy_2[train_choice_2]
    trainy_3 = trainy_3[train_choice_3]
    trainy_4 = trainy_4[train_choice_4]
    trainy_5 = trainy_5[train_choice_5]
    trainy_6 = trainy_6[train_choice_6]
    trainy_7 = trainy_7[train_choice_7]
    trainy_8 = trainy_8[train_choice_8]
    trainy_9 = trainy_9[train_choice_9]
    
    valX = np.concatenate((valX_0, valX_1, valX_2, valX_3, valX_4, valX_5, valX_6, valX_7, valX_8, valX_9), axis=0)
    valy = np.concatenate((valy_0, valy_1, valy_2, valy_3, valy_4, valy_5, valy_6, valy_7, valy_8, valy_9), axis=0)
    
    trainX = np.concatenate((trainX_0, trainX_1, trainX_2, trainX_3, trainX_4, trainX_5, trainX_6, trainX_7, trainX_8, trainX_9), axis=0)
    trainy = np.concatenate((trainy_0, trainy_1, trainy_2, trainy_3, trainy_4, trainy_5, trainy_6, trainy_7, trainy_8, trainy_9), axis=0)
    
    valX, valy = unison_shuffled_copies(valX, valy)
    trainX, trainy = unison_shuffled_copies(trainX, trainy)
    
    print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
    print('Validation: X=%s, y=%s' % (valX.shape, valy.shape))
    print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
    
    trainX = trainX.astype(np.float64)
    valX = valX.astype(np.float64)
    testX = testX.astype(np.float64)
    
    return trainX, trainy, valX, valy, testX, testy
    

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_cutted_data(valX, testX, trainX):
    cutted_valX = valX.copy()
    cutted_valX[:, :, 24:32, :] = 0
    
    cutted_testX = testX.copy()
    cutted_testX[:, :, 24:32, :] = 0
    
    cutted_trainX = trainX.copy()
    cutted_trainX[:, :, 24:32, :] = 0
    
    return cutted_valX, cutted_testX, cutted_trainX


def euclidean_distance_loss_control_nan(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    
    d = K.cast((y_pred - y_true), dtype='float32')
    
    loss = K.sum(K.square(d))
    
    division_num =K.cast(tf.shape(y_true)[0], dtype='float32')
    
    loss_division = loss / division_num
    
    lnLoss = tf.where(tf.math.is_inf(loss_division), tf.ones_like(loss_division), K.sqrt(loss_division))
    
    return lnLoss


def define_generator_encoder():
    momentum_value = 0.5
    
    encoder_input = Input(shape=(32, 32, 3,))
    
    ### Conv
    model = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(encoder_input)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Flatten()(model)
    model = Dense(16 * 16 * 2)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    generator_model = Model(inputs=[encoder_input], outputs=model)
    return generator_model


def define_generator_decoder():
    momentum_value = 0.5
    
    decoder_input = Input(shape=(16 * 16 * 2))
    
    model = Reshape([4, 4, 16 * 2])(decoder_input)
    
    ### TConv
    model = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    
    ### Conv
    model = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = Activation('tanh')(model)
    
    generator_model = Model(inputs=[decoder_input], outputs=model)
    return generator_model


def get_input_and_label_data(valX, cutted_valX, g_e_model):
    target_data = g_e_model.predict(valX)
    input_data = g_e_model.predict(cutted_valX)
    
    return input_data, target_data


def define_lstm_cnn_model():
    momentum_value = 0.5
    
    ##### Input
    input_cnn = Input(shape=(32, 32, 3))
    input_lstm = Input(shape=(16 * 16 * 2))
    
    ### cnn
    model = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(input_cnn)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model_out_cnn = Flatten()(model)
    
    ### lstm
    model = Reshape((32, 16))(input_lstm)
    
    model = Bidirectional(LSTM(512, kernel_initializer='he_normal'))(model)
    model_out_lstm = LeakyReLU(alpha=0.2)(model)
    
    
    ### concatenate
    model_middle = concatenate([model_out_cnn, model_out_lstm])
    
    ### CNN Output
    model = Reshape((64, 32, 1))(model_middle)
    
    model = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(512, (3, 3), strides=(2, 1), padding='same', kernel_initializer='he_normal')(model)
    model = BatchNormalization(momentum=momentum_value)(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Conv2D(3, (7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(model)
    model_cnn_out = Activation('tanh')(model)
    
    ### LSTM Output
    model = Reshape((128, 16))(model_middle)
    
    model = Bidirectional(LSTM(512, kernel_initializer='he_normal'))(model)
    model = LeakyReLU(alpha=0.2)(model)
    
    model = Dense(16 * 16 * 2)(model)
    model_out_lstm = LeakyReLU(alpha=0.2)(model)
    
    ### Model Config
    generator_model = Model(inputs=[input_cnn, input_lstm], outputs=[model_cnn_out, 
                                                                     model_out_lstm])
    generator_model.compile(loss='mse', optimizer=opt, 
                            metrics=['mse'])
    return generator_model


def summarize_performance(epoch, lstm_cnn_model, g_d_model, train_loss, 
                          cutted_trainX_full, train_input_lstm, trainX_full):
    filename_lstm_cnn= 'FeedBack/Model/lstm_cnn_model/lstm_cnn_model_%03d_%05f.h5' % (
        epoch, train_loss) 
    lstm_cnn_model.save(filename_lstm_cnn)
    
    image_number = 9
    
    rd_index = np.random.randint(0, cutted_trainX_full.shape[0], image_number)
    
    batch_input_cnn = cutted_trainX_full[rd_index]
    batch_input_lstm = train_input_lstm[rd_index]
    batch_target_cnn = trainX_full[rd_index]
    batch_target_lstm = train_target_lstm[rd_index]
    
    predictions = lstm_cnn_model.predict([batch_input_cnn, batch_input_lstm])
    
    fig, axes = plt.subplots(image_number, 4, figsize=(10,20))
    
    for img_num in range(image_number):
        x_output_cnn = np.array((batch_target_cnn[img_num] * 127.5 + 127.5)).astype(np.uint8)
        x_output_lstm = g_d_model.predict(batch_target_lstm[img_num:img_num+1])[0]
        x_prediction_cnn = np.array((predictions[0][img_num] * 127.5 + 127.5)).astype(np.uint8)
        x_prediction_lstm = predictions[1][img_num]
        
        x_prediction_lstm = x_prediction_lstm.reshape(1, x_prediction_lstm.shape[0])
        x_prediction_lstm = g_d_model.predict(x_prediction_lstm)[0]
        
        x_output_lstm = np.array((x_output_lstm * 127.5 + 127.5)).astype(np.uint8)
        x_prediction_lstm = np.array((x_prediction_lstm * 127.5 + 127.5)).astype(np.uint8)
        
        axes[img_num,0].imshow(x_output_cnn)
        axes[img_num,1].imshow(x_output_lstm)
        axes[img_num,2].imshow(x_prediction_cnn)
        axes[img_num,3].imshow(x_prediction_lstm)
    
    fig.suptitle('Target, Decoder Out, CNN Out, LSTM Out', fontsize=20)
    plt.savefig('FeedBack/Summarize/_' + str(epoch) + '.fig.png')


def model_train(lstm_cnn_model, cutted_trainX_full, train_input_lstm, trainX_full, 
                train_target_lstm, test_input_lstm, test_target_lstm, g_d_model,
                model_number, n_batch, n_epochs):
    bat_per_epo = int(cutted_trainX_full.shape[0] / n_batch)
    
    for i in range(model_number + 1, model_number + n_epochs + 1):
        train_losses_np = np.zeros(bat_per_epo)
        for j in range(bat_per_epo):
            rd_index = np.random.randint(0, cutted_trainX_full.shape[0], n_batch)
            
            batch_input_cnn = cutted_trainX_full[rd_index]
            batch_input_lstm = train_input_lstm[rd_index]
            batch_target_cnn = trainX_full[rd_index]
            batch_target_lstm = train_target_lstm[rd_index]
            
            ### Train lstm_cnn_model
            g_loss = lstm_cnn_model.train_on_batch([batch_input_cnn, 
                                                    batch_input_lstm], 
                                                   [batch_target_cnn,
                                                    batch_target_lstm])
            
            train_losses_np[j] = g_loss[0]
        print('>%d, train_mse_loss=%.5f'%(i, train_losses_np.mean()))
        if i % 10 == 0 or i == 1:
            summarize_performance(i, lstm_cnn_model, g_d_model, train_losses_np.mean(), 
                                  cutted_trainX_full, train_input_lstm, trainX_full)
            


if __name__ == "__main__":
    trainX, trainy, valX, valy, testX, testy = load_dataset()
    
    cutted_valX, cutted_testX, cutted_trainX = get_cutted_data(valX, testX, trainX)
    
    cutted_trainX_full = np.zeros((50000, 32, 32, 3))
    cutted_trainX_full[0:10000] = cutted_valX
    cutted_trainX_full[10000:] = cutted_trainX
    
    trainX_full = np.zeros((50000, 32, 32, 3))
    trainX_full[0:10000] = valX
    trainX_full[10000:] = trainX
    
    ### AutoEncoder Model
    g_e_model = define_generator_encoder()
    g_d_model = define_generator_decoder()
    
    model_number = 310
    
    try:
        g_e_model.load_weights("FeedBack/Model/generator_encoder_model_%03d.h5" % (model_number))
        g_d_model.load_weights("FeedBack/Model/generator_decoder_model_%03d.h5" % (model_number))
    except:
        print('Model Bulunamadı.')
        
    train_input_lstm, train_target_lstm = get_input_and_label_data(trainX_full, 
                                                                   cutted_trainX_full, 
                                                                   g_e_model)
    test_input_lstm, test_target_lstm = get_input_and_label_data(testX, cutted_testX, 
                                                                 g_e_model)
    
    # Model LSTM-CNN
    lstm_cnn_model = define_lstm_cnn_model()
    model_number = 1130
    try:
        lstm_cnn_model.load_weights("FeedBack/Model/lstm_cnn_model/lstm_cnn_model_%03d_0.010437.h5" % (model_number))
    except:
        print('lstm_cnn_model Model Bulunamadı.')
    
    # Train lstm_cnn_model
    model_train(lstm_cnn_model, cutted_trainX_full, train_input_lstm, trainX_full, 
                train_target_lstm, test_input_lstm, test_target_lstm, g_d_model,
                model_number, 32, 10000)