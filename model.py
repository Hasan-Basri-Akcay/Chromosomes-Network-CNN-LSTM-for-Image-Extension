import cv2
import sys
import numpy as np


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LSTM, Bidirectional


opt = Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt_d = Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
    if programPause == 'q':
        sys.exit()


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
    

class LstmCnnModel():
    
    def __init__(self, path_lstm, path_decoder):
        self.lstm_cnn_model = define_lstm_cnn_model()
        self.lstm_cnn_model.load_weights(path_lstm)
        
        self.g_d_model = define_generator_decoder()
        self.g_d_model.load_weights(path_decoder)
    
    
    def predict(self, data):
        predictions_lstm = data[0].copy()
        predictions_cnn = data[0].copy()
        
        preds = self.lstm_cnn_model.predict(data)
        
        preds_cnn = np.array(preds[0])
        preds_lstm = np.array(preds[1])
        
        predictions_cnn[:, :, 24:, :] = preds_cnn[:, :, 24:, :]
        preds_lstm_img = self.g_d_model.predict(preds_lstm)
        predictions_lstm[:, :, 24:, :] = preds_lstm_img[:, :, 24:, :]
        
        return [predictions_cnn, predictions_lstm]
