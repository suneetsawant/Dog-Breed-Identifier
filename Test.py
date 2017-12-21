

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.misc import imread,imsave
from PIL import Image
from keras.preprocessing import image



def Test(df,model_path,data_generator,batch_size,test_ids):
    tf.reset_default_graph()
    

    x_dim = 7*7*512

## Redefine the Graph  with final softmax layer 
    X = tf.placeholder(tf.float32, shape=(batch_size, x_dim))


    FC = tf.layers.dense(inputs=X,units=2048, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
    drop_FC = tf.nn.dropout(FC,keep_prob=1)

    FC2 = tf.layers.dense(inputs=drop_FC,units=2048, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
    drop_FC2 = tf.nn.dropout(FC2,keep_prob=1)

    y = tf.layers.dense(inputs= drop_FC2,units=120)

    pred = tf.nn.softmax(y)


    labels = df.columns
    out = pd.DataFrame(columns=labels)
    out['id'] = test_ids


## Run the model on test images 
    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        for i in range(1,len(test_ids)):
            x_batch,Id = next(data_generator)
            out.loc[out['id']==Id[0],'affenpinscher':] = pred.eval(feed_dict={X:x_batch})
            #if i%1000==0: print(i,)


## Delete the breed column
    del out['breed']


## Save the result file
    out.to_csv('result.csv',index=False)
    print("TEST RESULTS STORED IN  /result.csv")



