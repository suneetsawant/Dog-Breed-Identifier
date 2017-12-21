from utils import * 
from Train import *
from Test import * 
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
	'--train',  
	type=str,
	default='train/', 
    help=' Path to Training  Images')

parser.add_argument(
	'--test',  
	type=str,
	default='test/', 
    help=' Path to Test  Images')

parser.add_argument(
	'--iter',  
	type=int,
	default='20000', 
    help=' Number of training iterations')

parser.add_argument(
	'--lr',  
	type=float,
	default='0.001', 
    help=' Learning rate of Adam optimizer')

parser.add_argument(
	'--eps',  
	type=float,
	default='1', 
    help=' epsilon for optimizer')


args = parser.parse_args()


## Define the train and test images path
train_image_path = args.train
test_image_path = args.test


## Defining the path where features extracted from VGG19 are stored 
train_feature_path = 'features/train/'
test_feature_path = 'features/test/'
model_path = 'Model/'

## Number of iterations to train the model 
iterations = args.iter
learning_rate = args.lr
epsilon = args.eps

print('******************** LOADING IMAGES AND CREATING FEATURES ********************')
print('')

## Create features of training images

print("Creating features of training images in /%s"%train_image_path)
if not os.path.exists(train_feature_path):	
   os.makedirs(train_feature_path)
   create_features(train_image_path,train_feature_path)
else :
   create_features(train_image_path,train_feature_path)
print('')


## Create features of test images	
print("Creating features of test images in /%s"%test_image_path)
if not os.path.exists(test_feature_path):
   os.makedirs(test_feature_path)
   create_features(test_image_path,test_feature_path)
else:
   create_features(test_image_path,test_feature_path)

print('')


## Read the labes files and create one hot encoding of labels
df = pd.read_csv('labels.csv')
df2 = pd.get_dummies(df['breed'])
df = df.join(df2)

## Train the model with batchsize of 10 

batch_size = 10
train_ids = []
for filename in os.listdir(train_feature_path):
    train_ids.append(os.path.splitext(filename)[0])

data = DataGenerator(batch_size,train_feature_path,True).generate(df,train_ids)


print('')
print("******************** TRAINING MODEL ON IMAGES IN /"+ train_image_path+' *********************')
Train(df,model_path,data,batch_size,iterations,learning_rate,epsilon)

## Test the model one image at a time
test_ids = []
for filename in os.listdir(test_feature_path):
    test_ids.append(os.path.splitext(filename)[0]) 

batch_size = 1
data = DataGenerator(batch_size,test_feature_path,False).generate(df,test_ids)


print('')
print('*********************** TESTING MODEL ON IMAGES IN /'+ test_image_path+' ********************')

Test(df,model_path,data,batch_size,test_ids)
