'''
2018 Johannes Weirich
Module for use of tensorflow on image data using convolutional neural network	 
	Containing functions: 
        image_gallery: Showing a collection of named images in an array
	define_network: Defining convol. neural network structure for image recognintion
	assign names_probabilities: Creates a list of names and probabilities for the predictions
'''

from __future__ import division, print_function, absolute_import

from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from glob import glob

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.initializers import variance_scaling
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
#from sklearn.datasets import olivetti_faces
from tflearn.datasets import cifar10

###################################
# Image Gallery function
###################################
def image_gallery(imgs,names,cols=6,dt='uint8'):
  '''
  function for plotting a collection of images
  Parameters
  ----------
  imgs: rank 4 numpy array, shape=(N_images,size_x,size_y,3)
  names: list with image titles, shape=(N_images,)   
  '''
  rows=int(imgs.shape[0]/cols)+1
  fig,ax=plt.subplots(nrows=rows,ncols=cols)
  ctr=0
  for idx_row in range(rows): 
    for idx_col in range(cols):
      print('row, col, ctr',idx_row,idx_col)
      if ctr<imgs.shape[0]:
        ax[idx_row,idx_col].imshow(np.array(imgs[ctr],dtype=dt))
        ax[idx_row,idx_col].set_title(names[ctr],fontsize=9)
      
      ctr+=1
  
  plt.subplots_adjust(hspace=0.9)	
  plt.show()



###################################
### Import picture files 
###################################
def load_training_files(directories,size_img,extension='jpg'):
  '''
  function for loading files for training of cnn
  Parameters
  ----------
  directories: List with directories where files are stored
  size_img: image pixel size,
  extension: image type
  
  Returns
  -------
  allX, Numpy rank 4 array, shape (N_images, size_x,size_y,3)
  ally, Labels, list, shape (N_images,)
  '''
  n_files=0
  ##to_do: first for loop not necessary with all_values as list of tuples
  for directory in directories:
    path=os.path.join(directory,'*.'+extension)
    n_files+=len(sorted(glob(path)))

  #to_do: make list of tuples istead of 
  all_values=[]
  allX = np.zeros((n_files, size_img, size_img, 3),dtype=float)
  ally = np.zeros(n_files)
  count = 0
  for label,directory in enumerate(directories):
    file_path=os.path.join(directory,'*.'+extension)
    files=sorted(glob(file_path))
    for f in files:
      try:
        img = io.imread(f)
        if img.shape[0]!=size_img:
          img = imresize(img, (size_img, size_img, 3))
        all_values+=[(img,label)]
        allX[count] = img
        ally[count] = label
        count += 1
      except:
        continue
  
  return allX,ally
   

def define_network(size_img,N_classes):
  '''
  Function for defining a particular convolutional network including 
  3 convolution layers, and 2x max-pooling, as well as 2x fully connected
  Parameters
  ----------
  size_img: number of pixels of images to be used, with shape=(size_img,size_img)
  N_classes: The number of output classes/cathegories 
  '''
  ###################################
  # Image transformations
  ###################################

  # normalisation of images
  img_prep = ImagePreprocessing()
  img_prep.add_featurewise_zero_center()
  img_prep.add_featurewise_stdnorm()

  # Create extra synthetic training data by flipping & rotating images
  img_aug = ImageAugmentation()
  img_aug.add_random_flip_leftright()
  img_aug.add_random_rotation(max_angle=25.)

  ###################################
  # Define network architecture
  ###################################

  # Input is a size_imagexsize_image, 3 color channels (red, green and blue)
  network = input_data(shape=[None,size_img,size_img, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

  # 1: Convolution layer 32 filters, each 3x3x3
  conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

  # 2: Max pooling layer
  network = max_pool_2d(conv_1, 2)

  # 3: Convolution layer with 64 filters
  conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

  # 4: Once more...
  conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

  # 5: Max pooling layer
  network = max_pool_2d(conv_3, 2)

  # 6: Fully-connected 512 node layer
  network = fully_connected(network, 512, activation='relu')

  # 7: Dropout layer to combat overfitting
  network = dropout(network, 0.5)

  # 8: Fully-connected layer N_classes outputs
  network = fully_connected(network, N_classes, activation='softmax')

  # Configure of Network training
  acc = Accuracy(name="Accuracy")
  network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

  return network
  

def assign_names_probabilities(pred,input_names=['Felix','Julius','Other']):
  '''
  Function returning name corresponding to highest probability
  Parameters 
  ----------
  pred: numpy NxM array, with N number of predictions and M number of labels/names 
  input_names: list or numpy array of strings with length M
  
  Returns
  --------
  list of strings containing name with highest probability as well as probability
  '''
  names_prob=[]
  Y_pred=[]
  for vals in pred:
      names_prob+=[input_names[np.argmax(vals)]+' prob='+str(vals[np.argmax(vals)])]
      Y=np.zeros(len(vals))
      Y[np.argmax(vals)]=1
      Y_pred+=[Y]
  Y_pred=np.array(Y_pred)
  return names_prob

############################################################
#Automated testing section when module is called with 'run'
############################################################
if __name__=='__main__':

  #size_img=50
  checkpoint='Model_CP.tflearn'
  run_id='Model_CP'
  filename='Face_identification.tflearn'
  
  #base_path='/home/jowe/Pictures/2018-04-faces/'
  #subdirs=['Julius','Felix','Marie','Johannes','Other']
  #N_classes=len(subdirs)
  #directories=[os.path.join(base_path,subdir) for subdir in subdirs]
  #label_names=subdirs
	
  #allX,ally=load_training_files(directories,size_img)
  #ally=to_categorical(ally,N_classes)
  (X, Y), (X_test, Y_test) = cifar10.load_data()
  size_img=X.shape[1]
  N_classes=len(np.unique(Y))
  Y=to_categorical(Y,N_classes)
  Y_test=to_categorical(Y_test,N_classes)
  label_names=['Plane','Car','bird','cat','deer','dog','frog','horse','ship','truck'] 
  #X,X_test,Y,Y_test=train_test_split(allX,ally,test_size=0.2,random_state=7)
  network=define_network(size_img,N_classes)
  model = tflearn.DNN(network, checkpoint_path=checkpoint, max_checkpoints = 3, tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')
  ###################################
  # Train model for Nepochs
  ###################################
  
  Nepochs=20
  model.fit(X.copy(), Y.copy(), validation_set=(X_test, Y_test), batch_size=500,
      	   n_epoch=Nepochs, run_id=run_id, show_metric=True)

  model.save(filename)
  pred=model.predict(X_test.copy())
  names_prob=assign_names_probabilities(pred,label_names)
  image_gallery(X_test[0:20],names_prob[0:20],cols=5,dt='float')

