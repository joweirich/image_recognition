from sklearn.model_selection import train_test_split
import image_recognition as img_rec
import face_detect_extract as fde
from tflearn.data_utils import to_categorical
import tflearn
import numpy as np
import glob
import os
from PIL import Image
from scipy.misc import imresize
############################################
#Load image collection
############################################

folders=glob.glob('/home/jowe/Pictures/2018*')

family_path='/home/jowe/Pictures/Family-2017-For-Album/'
small_images_path=family_path+'faces'

#################################################
### Define and load neural network
#################################################
'''todo: Do we always have to define the network before loading?
        how to update network with new data
'''
#input_names=['Felix','Marie','Julius','Other']
input_names=['Felix','Julius','Marie','Johannes','Other']
nb_classes=len(input_names)
size_image=50
network=img_rec.define_network(size_image,nb_classes)
model = tflearn.DNN(network, checkpoint_path='model_Felix_Julius_Marie_Johannes.tflearn', max_checkpoints = 3, tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')

model.load('model_Felix_Julius_Marie_Johannes_final.tflearn')

#######################################################
#Load files in folders, check for faces and apply CNN 
#for face recognition of Felix,Julius,Marie
########################################################
resize=800
thumb_size=size_image
casc_path='./FaceDetect/haarcascade_frontalface_default.xml'
for folder in folders:
  print('Searching for faces in ',folder)
  files=glob.glob(os.path.join(folder,'*.JPG'))
  files+=glob.glob(os.path.join(folder,'*.jpg'))
  ##################################################
  ### Find faces 
  #################################################
  for fil in files:
    try:
      Img0=Image.open(fil)
      Img=fde.auto_rotate(Img0)
      if max(Img.size)>resize:
        Npxls_long=resize
        Npxls_short=int(min(Img.size)/max(Img.size)*resize)
        if Img.size[0]>Img.size[1]: NewShape=[Npxls_long,Npxls_short]
        else: NewShape=[Npxls_short,Npxls_long]

        Img=Img.resize(NewShape,Image.ANTIALIAS)
        faces=fde.find_faces(np.asarray(Img,dtype='uint8'),casc_path,scaleF=1.25,minNeighb=5,face_ratio=0.05,)
        if len(faces)>0:
          print('found {0} faces'.format(len(faces)))
          imgs_reduced=fde.export_face_area(Img,faces,fil,small_images_path,scaling=1.0,write_to_disk=True) #imgs_reduced is dict of PIL images
          #export_face_area returns 100x100 thumbs, should probably be changed into input parmeter
          thumbs=[]
          for img_key in imgs_reduced:
            img=np.asarray(imgs_reduced[img_key],dtype='float32')
            if thumb_size<img.shape[0]:
              img = imresize(img, (thumb_size, thumb_size, 3))
            thumbs+=[img] 
          thumbs=np.array(thumbs,dtype='float32') 
          pred=model.predict(thumbs.copy())
          names=img_rec.assign_names_probabilities(pred,input_names)
          for idx,pr in enumerate(pred):
            print(pred)
            print('found names',names)
            if pr.argmax()<4:
              print('Family image found, saving reduced image in',family_path)
              try:
                new_file_path=os.path.join(family_path,names[idx]+'.jpg')
                Img0.save(new_file_path) 
              except FileNotFoundError:
                print('Directory does not exist, face area not saved')
              except:
                print('Other Error, maybe wrong image scaling')

            
    except FileNotFoundError:
      print('File not found')




