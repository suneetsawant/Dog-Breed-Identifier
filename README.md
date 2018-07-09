# Dog Breed Identifier

This is an implemetation of image classifier to identify dog breeds using kaggle dataset [DOG BREED](https://www.kaggle.com/c/dog-breed-identification). 

<img src=output.jpg  hspace="20" alt="content" title="content" /> 


## Usage 
__python main.py --train train/ --test test/ --iter 50__

__Options__  
--train = path to training images  
--test  = path to test images  
--iter  = Number of iterations for training  
--lr    = Learning rate of optimizer  
--eps   = epsilon for optimizer

The folders train and test contains 50 sample dog images each and are used for sample run. The parameters are actually tuned for full scale dataset and should replace the sample images. 
