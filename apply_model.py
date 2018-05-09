from sklearn.model_selection import train_test_split
import image_recognition as img_rec
import face_detect_extract as fde
from tflearn.data_utils import to_categorical
import tflearn
import numpy as np
############################################
#Load image collection
############################################

base_path='/home/jowe/Pictures/2018-03_Phone_Transfer/'
base_path='/home/jowe/Pictures/2017-08-Galaxy-S6-Phone-Transfer/'

face_path=base_path+'faces'
img_coll=fde.load_img_coll(base_path)
nb_classes=4
size_image=100
##################################################
### Find faces and export thumbs to faces folder
#################################################
AllThumbs,filnames=fde.thumb_coll(img_coll,face_path)  

image_array=np.array([np.asarray(thumb,dtype='float32') for thumb in AllThumbs])

#################################################
### Define and load neural network
#################################################
'''todo: Do we always have to define the network before loading?
	how to update network with new data
'''
network=img_rec.define_network(size_image,nb_classes)
model = tflearn.DNN(network, checkpoint_path='model_Felix_Julius.tflearn', max_checkpoints = 3, tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

model.load('model_Felix_Julius_Marie.tflearn')
####################################
# Prediction on image array
####################################
pred=model.predict(image_array.copy())

#################################
# Show predictions in galery
##################################
'''todo: Make galery over more than one figure to keep overview'''
names=img_rec.assign_names_probabilities(pred,['Felix','Marie','Julius','Other'])
img_rec.image_galery(image_array[0:40],names[0:40])
