


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.misc import imread,imsave
from PIL import Image
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Input
from random import shuffle
import h5py

import os

model = VGG19(weights='imagenet',include_top=False)

def create_features(image_path,feature_path): 
    files = os.listdir(image_path)
    i = 0
    for filename in files:
        if i%10==0 and i!=0:print ('Reading of '+str(i)+' files completed')
        i += 1  
        path = feature_path+os.path.splitext(filename)[0]+'.feat'

        if os.path.isfile(path)==False:
            img = image.load_img(image_path+filename, target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(np.expand_dims(img, axis=0)) 
            res = np.ndarray.flatten(model.predict(img))
            np.savetxt(path,res,delimiter=',')
    print("Completed feature creation of %d images"%i) 
    print("Features Stored in /%s"%feature_path)       
          

class DataGenerator():
    def __init__(self,batchsize,feature_path,train=True):
        self.batchsize = batchsize
        self.feature_path = feature_path
        self.train = train
        
    def generate(self,df,ID):
        while 1 : 
            index = np.arange(len(ID))
            #np.random.shuffle(index)
            
            size = int(len(ID)/self.batchsize)

            for i in range(size):
                Id = [ID[x] for x in index[i*self.batchsize:(i+1)*self.batchsize] ]
                
                x,y = self.data_gen(df,Id)
                
                if self.train: yield x,y
                else : yield x,Id 

    def data_gen(self,df,ID):
        X = np.empty([self.batchsize,7*7*512])
        y = np.empty([self.batchsize,120])
        labels = df.columns.values[2:]
        for i,idx in enumerate(ID):
            path = self.feature_path+idx+'.feat'

            X[i] = np.loadtxt(path)
            
            if self.train:
                y[i] = df.loc[df['id']==idx,'affenpinscher':].values
            
                assert labels[np.where(y[i]==1)] == df.loc[df['id']==idx,'breed'].values
        return X,y    

def deprocess_image(x, img_nrows, img_ncols):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
# Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        #scipy.misc.imsave(path, x)
    return x 





