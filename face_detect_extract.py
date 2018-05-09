'''
2018 @Johannes Weirich
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
  found_ori_key=False
  for key, value in img._getexif().items():
    if TAGS.get(key) == 'Orientation':
      ori = value
      found_ori_key=True
      if ori==2: img=ImageOps.mirror(img)
      elif ori==3: img=ImageOps.mirror(ImageOps.flip(img))
      elif ori==4: img=ImageOps.flip(img)
      elif ori==5: img=ImgageOps.mirror(img.rotate(-90))
      elif ori==6: img=img.rotate(-90)
      elif ori==7: img=ImgageOps.mirror(img.rotate(90))
      elif ori==8: img=img.rotate(90)
    
  if not found_ori_key:
    print('could not extract orientation from metadata') 
  return img
    

def load_images(path,extension='jpg'):
  '''
  Loading all images with given extensions in folder and subfolder
  Parameters
  ------------
  path: Base directory, which will be searched for images including all subfolders
  extension: Image file extension

  Returns
  -------------
  ImgColl: Dict with PIL images 
  '''

  files_with_path=[]
    
  for dpth,c_dirnames,filenames in os.walk(path):
    files_with_path+=[os.path.join(dpth,fn) for fn in filenames 
		      if extension in fn]
  ImgColl=dict()

  for fil in files_with_path:
    print('Loading file:',os.path.basename(fil))
    try:
      Img=Image.open(fil)
      ImgColl[os.path.basename(fil)]=auto_rotate(Img)
      Img.close()
    except FileNotFoundError:
      print('File not found')
  
  return(ImgColl)


def resize_image(img,new_size):
  '''
  Scaling of image
  Parameters
  ------------
  img: PIL Image to be rescaled 
  new_size: size of the rescaleds image long side

  Returns
  -------------
  img_r: resized image
  '''

  if max(img.size)>new_size: 
    Npxls_long=new_size
    Npxls_short=int(min(img.size)/max(img.size)*new_size)
  
    if img.size[0]>img.size[1]: NewShape=[Npxls_long,Npxls_short]
    else: NewShape=[Npxls_short,Npxls_long]
  
    img=img.resize(NewShape,Image.ANTIALIAS)
  
  return img
  

def load_img_coll(path,extension='jpg',new_size=800):
  '''
  Loading all images with given extensions in specified folder,
  autorotate images and resize to given size
  
  Parameters
  ------------
  path: Directory where images are located 
  extension: Image file extension
  resize: Number of pixels for image resizing (

  Returns
  -------------
  ImgColl: Dict with PIL images 
  '''

  filenames=[fn for fn in os.listdir(path) if extension in fn]
  img_coll=dict()

  for fil in filenames:
    print('Loading file:',fil)
    fullfil=os.path.join(path,fil)
    try:
       img=Image.open(fullfil)
       img_r=auto_rotate(img)
       #img.close()
       img_r=resize_image(img_r,new_size)

       img_coll[os.path.basename(fil)]=img_r

    except FileNotFoundError:
      print('File not found')
    except:
      print('Something else went wrong, file not loaded')

  return(img_coll)


def find_faces(image,cascPath,scaleF=1.3,minNeighb=5,face_ratio=0.1):
  ''' 
   Face detection in image
   Parameters
   ----------
   Image: numpy array, shape (NyPixels,NxPixels,3)
   casPath: Full path to XML file defining the method for face detection
   scaleF: scaling step size for face detection, reduction at each image scale
    minNeighb: Determining Quality of detected faces
   face_ratio: Min. relative size of faces (in pixel) with respect to small side of image
   
  Returns
  -------
  faces: Corrdinate of lower left corner of faces frame as well as width and height
  '''
  
  # Create the haar cascade
  faceCascade = cv2.CascadeClassifier(cascPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert image to grayscale, shape (NyPixels,NxPixels)
  Npixels_small=min(gray.shape) #Number of pixels for smaller side of image for minSize  

  print('Shape of image',gray.shape)
  min_size=(int(Npixels_small*face_ratio), int(Npixels_small*face_ratio))
  faces = faceCascade.detectMultiScale(gray,scaleFactor=scaleF,
				       minNeighbors=minNeighb,
                                       minSize=min_size,
                                       flags = cv2.CASCADE_SCALE_IMAGE
    					#outputRejectLevels = True
    					#flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    					)
  return faces 

def plot_img_with_faces(img,faces):
  '''
  plotting image with frames on faces
  Parameters:
  -----------
  img: PIL image or np.array
  faces: face coordinates
  
  Returns:
  --------
  None
  '''
  fig, ax = plt.subplots()
  ax.imshow(img)
  for (x, y, w, h) in faces:
    # ax.plot(np.arange(x,x+w+1),y*np.ones(w+1),c='g')
    ax.plot([x,x+w],2*[y],'g')
    # ax.plot(np.arange(x,x+w+1),(y+h)*np.ones(w+1),c='g')
    ax.plot([x,x+w],2*[y+h],'g')
  
  plt.show()   

def export_face_area(img,faces,size_img=100,filename='faces.jpg',face_img_path='./',scaling=1.0,write_to_disk=True):
  '''
  function returning images of face area given input image and face pos. data. Writing 
  face images to disk 
  Parameters:
  -----------
  img: PIL image where faces should be found 
  faces: face coordinates
  size_img: new pixel size for face area images, in order to make thumbnails of same size  
  filename: filename of image, working as base_filename for face images
  scaling: 
  write_to_disk: True if face images should be saved to disk
  
  Returns:
  --------
  imgs_reduced: collection of face images found in 
  '''
  
  imgs_reduced=dict()

  for idx_face,face in enumerate(faces):
    x,width=face[0],faces[:,2].max() #use max width of all found face images
    y,height=face[1],faces[:,3].max() #use max height of all found face images
    xmin,xmax=x-int(width*(scaling-1.0)/2.0),x+int(width*(1+(scaling-1.0)/2.0))
    ymin,ymax=y-int(height*(scaling-1.0)/2.0),y+int(height*(1+(scaling-1.0)/2.0))
    #imgs_reduced[idx_img]=img[ymin:ymax,xmin:xmax]
    box=(xmin,ymin,xmax,ymax)#box for crop: xmin,ymin,xmax,ymax
    imgs_reduced[idx_face]=ImageOps.fit(img.crop(box),(size_img,size_img),Image.ANTIALIAS)
    fformat=os.path.splitext(filename)[1]
    fname=os.path.join(face_img_path,os.path.splitext(filename)[0]+'_f'+str(idx_face)+fformat)
    print(fname)
    if write_to_disk:
      try:
        #mpimg.imsave(fname,imgs_reduced[idx_img],format=fformat.split('.')[1])
        imgs_reduced[idx_face].save(fname)
      except FileNotFoundError:
        print('Directory does not exist, face area not saved')
      #except:
       # print('Other Error, maybe wrong image scaling')

  return imgs_reduced

def thumb_coll(ImgColl,thumb_path,CascPath='./FaceDetect/haarcascade_frontalface_default.xml',cv2params=[1.05,5,0.09]):
  '''
  Function creating list of thumbnails from image collection
  Parameters:
  -----------
  ImgColl: image collection, dict with keys=filenames and values=PIL images 
  thumb_path: path where thumbnails should be stored
  CascPath: full path to XML file defining the face detection method
  cv2params: parameters for face detection
  Returns:
  ========
  AllThumbs: numpy array shape (Nthumbs,xPixels,yPixels,3) 
  names: list of names of images where faces were found
  '''
  AllThumbs=[]
  names=[]
  for img_name in ImgColl:
    img=ImgColl[img_name]
    #find all faces for given image
    faces=find_faces(np.asarray(img,dtype='uint8'),CascPath,scaleF=cv2params[0],minNeighb=cv2params[1],face_ratio=cv2params[2])
    print('found {0} faces'.format(len(faces)))
    if len(faces)>0:
      imgs_reduced=export_face_area(img,faces,filename=img_name,face_img_path=thumb_path,scaling=1.5)
      names+=[img_name]
      for img_red in imgs_reduced:
        AllThumbs+=[(imgs_reduced[img_red])] #append all thumbnails to list

  return AllThumbs,names
 

############################################################
#Automated testing section when module is called with 'run'
############################################################
if __name__=='__main__':
  img_path='./'
  face_img_path=os.path.join(img_path,'faces')
  img_coll=load_img_coll(img_path,extension='JPG')
  allthumbs=thumb_coll(img_coll,face_img_path)
 
