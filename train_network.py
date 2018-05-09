from sklearn.model_selection import train_test_split
import image_recognition as img_rec
from tflearn.data_utils import to_categorical
import tflearn
############################################
#Training directories and corresponding names
############################################

base_path='/home/jowe/Pictures/2018-04-faces/'
Felix_path=base_path+'Felix'
Julius_path=base_path+'Julius'
Marie_path=base_path+'Marie'
Johannes_path=base_path+'Johannes'
Other_path=base_path+'Other'

directories=[Felix_path,Julius_path,Marie_path,Johannes_path,Other_path]
nb_classes=len(directories)

###################################
### Import picture files 
###################################
size_image=50
allX,ally=img_rec.load_training_files(directories,size_image)

###################################
# Prepare train & test samples
###################################

X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

Y = to_categorical(Y,nb_classes )
Y_test = to_categorical(Y_test,nb_classes)

####################################
#Define convolutional neural network
#####################################

network=img_rec.define_network(size_image,nb_classes)

####################################
#Define and train DNN model
#####################################

Nepochs=110

model = tflearn.DNN(network, checkpoint_path='model_Felix_Julius_Marie_Johannes.tflearn', max_checkpoints = 3, tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

model.fit(X,Y,validation_set=(X_test,Y_test),batch_size=500,n_epoch=Nepochs,run_id='Model_Felix_Julius_Marie_Johannes',show_metric=True)

model.save('model_Felix_Julius_Marie_Johannes_final.tflearn')
####################################
# Prediction on test data
####################################
pred=model.predict(X_test.copy())

#################################
# Show predictions in galery
##################################
names=img_rec.assign_names_probabilities(pred,['Felix','Julius','Marie','Johannes','Other'])
img_rec.image_galery(X_test[0:40],names[0:40])
