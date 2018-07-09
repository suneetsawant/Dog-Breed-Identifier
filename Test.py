

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.misc import imread,imsave
from PIL import Image
from keras.preprocessing import image
import cv2 


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
## Delete the breed column
    del out['breed']


## Run the model on test images 
    with tf.Session() as sess:
        saver = tf.train.Saver()

        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        for i in range(1,len(test_ids)):
            x_batch,Id = next(data_generator)
            out.loc[out['id']==Id[0],'affenpinscher':] = pred.eval(feed_dict={X:x_batch})
            breed_index = out.loc[out['id']==Id[0],'affenpinscher':].values.argmax(axis=1)+2
            breed = df.columns.values[breed_index]
            #print(df.columns.values.size,df.columns.values[breed+2])
            print('test/'+Id[0]+'.jpg')
            img = cv2.imread('test/'+Id[0]+'.jpg')
            
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (60,200)
            fontScale              = 0.9
            fontColor              = (0,255,0)
            lineType               = 2

            cv2.putText(img,breed[0], 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            #cv2.imshow(Id[0],img)
            cv2.imwrite('output/'+Id[0]+'.jpg',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



## Save the result file
    out.to_csv('result.csv',index=False)
    print("TEST RESULTS STORED IN  /result.csv")



