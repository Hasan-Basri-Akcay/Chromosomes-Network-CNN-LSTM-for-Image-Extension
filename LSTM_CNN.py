import sys
import cv2
import os
import numpy as np
import math
import pandas as pd
from scipy.linalg import sqrtm
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        ### Bildirim Seviyesi

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import tensorflow.keras.backend as K

import warnings

from manuel_models import MeanModel, MirrorModel, RepeatModel
from model import LstmCnnModel


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


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def ssim_score(original, compressed):
    y_true = K.cast(original, dtype='float32')
    y_pred = K.cast(compressed, dtype='float32')
    
    ssim_ms = tf.image.ssim_multiscale(y_true, y_pred, max_val=2, filter_size=2)
    
    ssim_ms = tf.math.reduce_mean(ssim_ms)
    
    return ssim_ms


def calculate_fid(act1, act2):
    act1 = act1.reshape(act1.shape[0], -1)
    act2 = act2.reshape(act2.shape[0], -1)
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
    

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


def summarize_test(index, n_batch, test_input, test_target, lstm_cnn_predictions, 
                   mean_predictions, repeat_predictions, mirror_predictions):
    
    fig, axes = plt.subplots(n_batch, 7, figsize=(10,20))
    
    for img_num in range(n_batch):
        if img_num == 0:
            axes[img_num,0].set_title('Input', fontsize=20)
            axes[img_num,1].set_title('Mean', fontsize=20)
            axes[img_num,2].set_title('Repeat', fontsize=20)
            axes[img_num,3].set_title('Mirror', fontsize=20)
            axes[img_num,4].set_title('OursLSTM', fontsize=20)
            axes[img_num,5].set_title('OursCNN', fontsize=20)
            axes[img_num,6].set_title('Target', fontsize=20)
        
        input_data = test_input[img_num]
        mean_pred = mean_predictions[img_num]
        repeat_pred = repeat_predictions[img_num]
        mirror_pred = mirror_predictions[img_num]
        lstm_pred = lstm_cnn_predictions[1][img_num]
        cnn_pred = lstm_cnn_predictions[0][img_num]
        test_target_data = test_target[img_num]
        
        axes[img_num,0].imshow(input_data)
        axes[img_num,1].imshow(mean_pred)
        axes[img_num,2].imshow(repeat_pred)
        axes[img_num,3].imshow(mirror_pred)
        axes[img_num,4].imshow(lstm_pred)
        axes[img_num,5].imshow(cnn_pred)
        axes[img_num,6].imshow(test_target_data)
    
    #fig.suptitle('Input, Mean, Repeat, Mirror, LSTM, CNN, Target', fontsize=20)
    plt.savefig('test best best/_' + str(index) + '.fig.png')


def model_test_plot(lstm_cnn_model, test_input_lstm, test_target_lstm, cutted_testX, testX,
                    g_d_model):
    n_batch = 9
    bat_per_epo = int(cutted_testX.shape[0] / n_batch)
    
    for i in range(bat_per_epo):
        rd_index = np.random.randint(0, cutted_testX.shape[0], n_batch)
        
        batch_input_cnn = cutted_testX[rd_index]
        batch_input_lstm = test_input_lstm[rd_index]
        batch_target_cnn = testX[rd_index]
        batch_target_lstm = test_target_lstm[rd_index]
        
        ### LSTM-CNN Model
        lstm_cnn_predictions = lstm_cnn_model.predict([batch_input_cnn, 
                                                       batch_input_lstm])
        
        ### Mean Model
        mean_predictions = MeanModel.predict(batch_input_cnn)
        
        ### Mirror Model
        mirror_predictions = MirrorModel.predict(batch_input_cnn)
        
        ### Repeat Last Pixel
        repeat_predictions = RepeatModel.predict(batch_input_cnn)
        
        summarize_test(i, n_batch, batch_input_cnn, batch_target_cnn, lstm_cnn_predictions, 
                      mean_predictions, repeat_predictions, mirror_predictions)


def model_test_score(lstm_cnn_model, test_input_lstm, cutted_testX, testX):
    ### Predictions
    # LSTM-CNN Model
    lstm_cnn_predictions = lstm_cnn_model.predict([cutted_testX, 
                                                   test_input_lstm])
    
    cnn_predictions = lstm_cnn_predictions[0]
    lstm_predictions = lstm_cnn_predictions[1]
    
    # Mean Model
    mean_predictions = MeanModel.predict(cutted_testX)
    
    # Mirror Model
    mirror_predictions = MirrorModel.predict(cutted_testX)
    
    # Repeat Last Pixel
    repeat_predictions = RepeatModel.predict(cutted_testX)
    
    
    ### Test Scores
    # RMSE
    rmse_mean = np.sqrt(np.mean((mean_predictions-testX)**2))
    rmse_repeat = np.sqrt(np.mean((repeat_predictions-testX)**2))
    rmse_mirror = np.sqrt(np.mean((mirror_predictions-testX)**2))
    rmse_cnn = np.sqrt(np.mean((cnn_predictions-testX)**2))
    rmse_lstm = np.sqrt(np.mean((lstm_predictions-testX)**2))
    
    # PSNR
    psnr_mean = PSNR(mean_predictions, testX)
    psnr_repeat = PSNR(repeat_predictions, testX)
    psnr_mirror = PSNR(mirror_predictions, testX)
    psnr_cnn = PSNR(cnn_predictions, testX)
    psnr_lstm = PSNR(lstm_predictions, testX)
    
    # SSIM
    ssim_mean = ssim_score(mean_predictions, testX).numpy()
    ssim_repeat = ssim_score(repeat_predictions, testX).numpy()
    ssim_mirror = ssim_score(mirror_predictions, testX).numpy()
    ssim_cnn = ssim_score(cnn_predictions, testX).numpy()
    ssim_lstm = ssim_score(lstm_predictions, testX).numpy()
    
    # FID
    fid_mean = calculate_fid(testX, mean_predictions)
    fid_repeat = calculate_fid(testX, repeat_predictions)
    fid_mirror = calculate_fid(testX, mirror_predictions)
    fid_cnn = calculate_fid(testX, cnn_predictions)
    fid_lstm = calculate_fid(testX, lstm_predictions)
    
    print('PSNR: ', psnr_mean, psnr_repeat, psnr_mirror, psnr_cnn, psnr_lstm)
    print('FID: ', fid_mean, fid_repeat, fid_mirror, fid_cnn, fid_lstm)
    
    scores_df = pd.DataFrame(columns=['Score', 'Mean', 'Repeat', 'Mirror', 
                                      'Ours CNN', 'Ours LSTM'])
    scores_df.loc[0, :] = ['RMSE', rmse_mean, rmse_repeat, rmse_mirror,
                           rmse_cnn, rmse_lstm]
    scores_df.loc[1, :] = ['PSNR', psnr_mean, psnr_repeat, psnr_mirror,
                           psnr_cnn, psnr_lstm]
    scores_df.loc[2, :] = ['SSIM', ssim_mean, ssim_repeat, ssim_mirror,
                           ssim_cnn, ssim_lstm]
    scores_df.loc[3, :] = ['FID', fid_mean, fid_repeat, fid_mirror,
                           fid_cnn, fid_lstm]
    
    scores_df[:] = scores_df[:].astype('object')
    scores_df.to_csv('scores.csv', sep=';', decimal='.', index=False)
    

def model_test_best_plot(lstm_cnn_model, test_input_lstm, test_target_lstm, 
                         cutted_testX, testX):
    lstm_cnn_predictions = lstm_cnn_model.predict([cutted_testX, 
                                                   test_input_lstm])
    
    lstm_predictions = lstm_cnn_predictions[1]
    rmse_lstm = np.mean((lstm_predictions-testX)**2, axis=(1, 2, 3))
    rmse_lstm_sort_index = np.argsort(rmse_lstm)
    
    repeat_predictions = RepeatModel.predict(cutted_testX)
    rmse_lstm_repeat = np.mean((lstm_predictions-repeat_predictions)**2, axis=(1, 2, 3))
    rmse_lstm_repeat_index = np.argsort(rmse_lstm_repeat)
    rmse_lstm_repeat_index = rmse_lstm_repeat_index[::-1]
    
    same_index = np.intersect1d(rmse_lstm_sort_index[:1000], 
                                rmse_lstm_repeat_index[:5000])
    
    # print(rmse_lstm_sort_index[:10])
    # print(rmse_lstm_repeat_index[:10])
    # print(len(same_index), same_index[:10])
    # pause()
    
    n_batch = 9
    img_num = 5
    
    for i in range(0, img_num):
        rd_index = same_index[i*n_batch:(i+1)*n_batch]
        
        batch_input_cnn = cutted_testX[rd_index]
        batch_input_lstm = test_input_lstm[rd_index]
        batch_target_cnn = testX[rd_index],
        batch_target_cnn = np.array(batch_target_cnn)
        
        ### LSTM-CNN Model
        lstm_cnn_predictions = lstm_cnn_model.predict([batch_input_cnn, 
                                                       batch_input_lstm])
        
        ### Mean Model
        mean_predictions = MeanModel.predict(batch_input_cnn)
        
        ### Mirror Model
        mirror_predictions = MirrorModel.predict(batch_input_cnn)
        
        ### Repeat Last Pixel
        repeat_predictions = RepeatModel.predict(batch_input_cnn)
        
        summarize_test(i, n_batch, batch_input_cnn, batch_target_cnn[0], 
                       lstm_cnn_predictions, mean_predictions, 
                       repeat_predictions, mirror_predictions)
        


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
    
    model_autoencoder_number = 310
    
    try:
        g_e_model.load_weights("FeedBack/Model/generator_encoder_model_%03d.h5" % (model_autoencoder_number))
        g_d_model.load_weights("FeedBack/Model/generator_decoder_model_%03d.h5" % (model_autoencoder_number))
    except:
        print('Model Bulunamadı.')
        
    train_input_lstm, train_target_lstm = get_input_and_label_data(trainX_full, 
                                                                   cutted_trainX_full, 
                                                                   g_e_model)
    test_input_lstm, test_target_lstm = get_input_and_label_data(testX, cutted_testX, 
                                                                 g_e_model)
    
    # Model LSTM-CNN
    model_number = 1600
    g_d_model_path = "FeedBack/Model/generator_decoder_model_%03d.h5" % (model_autoencoder_number)
    lstm_cnn_model_path = "FeedBack/Model/lstm_cnn_model/lstm_cnn_model_%03d_0.009558.h5" % (model_number)
    lstm_cnn_model = LstmCnnModel(lstm_cnn_model_path, g_d_model_path)
    
    # Train lstm_cnn_model
    # model_test_plot(lstm_cnn_model, test_input_lstm, test_target_lstm, cutted_testX, testX,
    #                 g_d_model)
    # model_test_score(lstm_cnn_model, test_input_lstm, cutted_testX, testX)
    model_test_best_plot(lstm_cnn_model, test_input_lstm, test_target_lstm, 
                         cutted_testX, testX)