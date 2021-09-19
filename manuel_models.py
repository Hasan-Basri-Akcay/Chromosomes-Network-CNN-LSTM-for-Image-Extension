import cv2
import sys
import numpy as np


def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
    if programPause == 'q':
        sys.exit()


class MeanModel():
    
    def predict(data):
        predictions = data.copy()
        for i in range(data.shape[0]):
            channel_0_mean = data[i, :, :, 0].mean()
            channel_1_mean = data[i, :, :, 1].mean()
            channel_2_mean = data[i, :, :, 2].mean()
            
            predictions[i, :, 24:, 0] = channel_0_mean
            predictions[i, :, 24:, 1] = channel_1_mean
            predictions[i, :, 24:, 2] = channel_2_mean
        return predictions


class MirrorModel():
    
    def predict(data):
        predictions = data.copy()
        for i in range(data.shape[0]):
            cut_part = data[i, :, 16:24, :]
            cut_part = cut_part[:,::-1,:]
            
            predictions[i, :, 24:, :] = cut_part
        return predictions


class RepeatModel():
    
    def predict(data):
        predictions = data.copy()
        for i in range(data.shape[0]):
            repeat_pixel = data[i, :, 23:24, :]
            repeat_img = np.concatenate((repeat_pixel, repeat_pixel, 
                                         repeat_pixel, repeat_pixel,
                                         repeat_pixel, repeat_pixel, 
                                         repeat_pixel, repeat_pixel), axis=1)
            
            predictions[i, :, 24:, :] = repeat_img
        return predictions