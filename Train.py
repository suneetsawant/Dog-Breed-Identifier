
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import re
from utils import * 


def Train(df,model_path,data_generator,batch_size,iterations,learning_rate,eps):

    x_dim = 7*7*512 ## Dimensions of the VGG19 model last layer without fully connected layers  
    y_dim = 120   ## Number of classes


## Defing the Graph 
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, shape=(batch_size, x_dim),name='X')
        Y = tf.placeholder(tf.float32, shape=(batch_size, y_dim),name='Y')


    with tf.name_scope('Fc1'):
        FC = tf.layers.dense(inputs=X,units=2048, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        drop_FC = tf.nn.dropout(FC,keep_prob=0.8)

    with tf.name_scope('Fc2'):
        FC2 = tf.layers.dense(inputs=drop_FC,units=2048, activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        drop_FC2 = tf.nn.dropout(FC2,keep_prob=0.8)

    with tf.name_scope('y'):
        y = tf.layers.dense(inputs= drop_FC2,units=120)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y))
    tf.summary.scalar('cost', cost)
        
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon = eps).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y,1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


## Running the optimization 
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=4)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('summaries',sess.graph)
        if not os.path.exists(model_path): 
            os.mkdir(model_path)
            sess.run(tf.global_variables_initializer())  
        else : 
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            print('Restoring model file from /'+model_path)
        
        for i in range(iterations):
            #s = time.time()
            x_batch,y_batch = next(data_generator)
            _,loss,acc,summary = sess.run([train_step,cost,accuracy,merged],feed_dict={X: x_batch, Y:y_batch})
            train_writer.add_summary(summary, i)
            
            if (i+1)%10==0 : 
                print("ITERATIONS= %d LOSS= %f ACCURACY= %f"%(i+1,loss,acc))
                save_path = saver.save(sess, model_path+'model.ckpt',global_step=i)
            #e = time.time()
            #print(e-s)   

        save_path = saver.save(sess, model_path+'model.ckpt')
    
        print("Model saved in file: %s" % save_path)

