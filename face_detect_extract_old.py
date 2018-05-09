'''
Module for importing images and doing operations on collections of images
Containing functions for 
	autorotation
	Importing and Transforming (reduction of size)
	Face detection and export of face areas
	Display images with face frames
	Exporting of face areas
'''

##################################################################################
# Module import, python does import modules from loaded modules cache
# 		 Hence no double import of modules. Use reload to 'refresh' modules  
##################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
import os
import pdb
import sys
#import tty
#tty.setcbreak(sys.stdin) #read from stdin (keyboard) without enter to confirm

###########################################
#Using lowercase _ separated function names
###########################################
def auto_rotate(img):
  '''
  Returns autorotated image based on exif (meta) information in jpgs
  Parameters
  ------------
  img: PIL Image

  Returns
  -------------
  img: Auto rotated PIL image 
  '''
  exifdict=img._getexif()
  metadict=dict()
  try:
    for k in exifdict.keys():
      if k in TAGS.keys():
        #print(TAGS[k],exifdict[k])
        metadict[TAGS[k]]=exifdict[k]
   
    if metadict['Orientation']: 
      ori=metadict['Orientation']
      if ori==2: img=ImageOps.mirror(img)
      elif ori==3: img=ImageOps.mirror(ImageOps.flip(img))
      elif ori==4: img=ImageOps.flip(img)
      elif ori==5: img=ImgageOps.mirror(img.rotate(-90))
      elif ori==6: img=img.rotate(-90)
      elif ori==7: img=ImgageOps.mirror(img.rotate(90))
      elif ori==8: img=img.rotate(90)
  except:
    print('something went wrong, image not autorotated') 
  return img
    

def load_images(path,extension='jpg'):
  files_with_path=[]
  if path[-1]!='/': path+='/' 
    
  for dpth,c_dirnames,filenames in os.walk(path):
    ###CONSIDER USE os.listdir() instead
    files_with_path=files_with_path+[dpth+fn for fn in filenames if extension in fn]
    #break #only filenames from specified subdirectory
  ImgColl=dict()
  #files_no_path=[fil.split('/')[-1] for fil in files_with_path]
  for idx_fil,fil in enumerate(files_with_path):
    print('Loading file:',fil.split('/')[-1])
    try:
      Img=Image.open(fil)
      ImgColl[fil.split('/')[-1]]=auto_rotate(Img)
    except FileNotFoundError:
      print('File not found')
  
  return(ImgColl)

def load_img_coll(path,extension='jpg',resize=800):
  if path[-1]!='/': path+='/'

  filenames=os.listdir(path)
  files_with_path=[path+fn for fn in filenames if extension in fn]
  #print(files_with_path)
  ImgColl=dict()
  #files_no_path=[fil.split('/')[-1] for fil in files_with_path]
  for idx_fil,fil in enumerate(files_with_path):
    print('Loading file:',fil.split('/')[-1])
    try:
      Img=Image.open(fil)
      Img=auto_rotate(Img)
      if max(Img.size)>resize:
        Npxls_long=resize
        Npxls_short=int(min(Img.size)/max(Img.size)*resize)
        if Img.size[0]>Img.size[1]: NewShape=[Npxls_long,Npxls_short]
        else: NewShape=[Npxls_short,Npxls_long]  
        Img=Img.resize(NewShape,Image.ANTIALIAS)

      ImgColl[fil.split('/')[-1]]=Img

    except FileNotFoundError:
      print('File not found')

  return(ImgColl)


def find_faces(image,cascPath,scaleF=1.3,minNeighb=5,face_ratio=0.1):
  # function for face detection in image
  # input: Image: array, shape (NyPixels,NxPixels,3)
  #	   casPath: Full path to XML file defining the method for face detection
  #        face_ratio: Relative size of faces (in pixel) with respect to small side of image
  
  # Create the haar cascade
  faceCascade = cv2.CascadeClassifier(cascPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert image to grayscale, shape (NyPixels,NxPixels)
  Npixels_small=min(gray.shape) #Number of pixels for smaller side of image for minSize  
  #pdb.set_trace()  #gives segmentation fault
  print('Shape of image',gray.shape)
  faces = faceCascade.detectMultiScale(gray,scaleFactor=scaleF,minNeighbors=minNeighb,
    minSize=(int(Npixels_small*face_ratio), int(Npixels_small*face_ratio)),
    flags = cv2.CASCADE_SCALE_IMAGE
    #outputRejectLevels = True
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    )
  return faces 

def plot_img_with_faces(img,faces):
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  ax.imshow(img)
  for (x, y, w, h) in faces:
    ax.plot(np.arange(x,x+w+1),y*np.ones(w+1),c='g')
    ax.plot(np.arange(x,x+w+1),(y+h)*np.ones(w+1),c='g')
  plt.show()   

def export_face_area(img,faces,filename,SmallImgPath,scaling=1.0,write_to_disk=True):
  #function returning face areas given input image and face pos. data
  imgs_reduced=dict()
  if SmallImgPath[-1]!='/': SmallImgPath+='/'
  for idx_img in range(faces.shape[0]):
    x,width=faces[idx_img,0],faces[:,2].max()
    y,height=faces[idx_img,1],faces[:,3].max()
    xmin,xmax=x-int(width*(scaling-1.0)/2.0),x+int(width*(1+(scaling-1.0)/2.0))
    ymin,ymax=y-int(height*(scaling-1.0)/2.0),y+int(height*(1+(scaling-1.0)/2.0))
    #imgs_reduced[idx_img]=img[ymin:ymax,xmin:xmax]
    box=(xmin,ymin,xmax,ymax)#box for crop: xmin,ymin,xmax,ymax
    imgs_reduced[idx_img]=ImageOps.fit(img.crop(box),(100,100),Image.ANTIALIAS)
    fformat=os.path.splitext(filename)[1]
    fname=SmallImgPath+os.path.splitext(filename.split('/')[-1])[0]+'_f'+str(idx_img)+fformat
    print(fname)
    if write_to_disk:
      try:
        #mpimg.imsave(fname,imgs_reduced[idx_img],format=fformat.split('.')[1])
        imgs_reduced[idx_img].save(fname)
      except FileNotFoundError:
        print('Directory does not exist, face area not saved')
      except:
        print('Other Error, maybe wrong image scaling')

  return imgs_reduced

def thumb_coll(ImgColl,FacesPath,CascPath='./FaceDetect/haarcascade_frontalface_default.xml',cv2params=[1.3,2,0.09]):
  #Function creating list of thumbnails from image collection
  #Input image collection, dict with keys=filenames and values=PIL images 
  #Output numpy array shape (Nthumbs,xPixels,yPixels,3), list names
  AllThumbs=[]
  names=[]
  for img_name in ImgColl:
    img=ImgColl[img_name]
    #find all faces for given image
    faces=find_faces(np.asarray(img,dtype='uint8'),CascPath,scaleF=cv2params[0],minNeighb=cv2params[1],face_ratio=cv2params[2])
    print('found {0} faces'.format(len(faces)))
    if len(faces)>0:
      imgs_reduced=export_face_area(img,faces,img_name,FacesPath,1.5)
      names+=[img_name]
      for img_red in imgs_reduced:
        AllThumbs+=[(imgs_reduced[img_red])] #append all thumbnails to list

  return AllThumbs,names
 

############################################################
#Automated testing section when module is called with 'run'
############################################################
if __name__=='__main__':
  img_path='/home/jowe/Pictures/'
  faces_path=img_path+'faces'
  img_coll=load_img_coll(img_path,extension='jpg')
  thumb_coll=thumb_coll(img_coll,FacesPath)
 
