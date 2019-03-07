#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:21:58 2019

@author: Jan Lause - https://github.com/jlause
"""

import h5py
import numpy as np
import PIL
import subprocess
import os

def run_unet_segmentation(input_img,img_pixelsize_x,img_pixelsize_y,
                          modelfile_path,weightfile_path,iofile_path,
                          tiling_x=228,tiling_y=228,gpu_flag='',
                          cleanup=True):
    '''Applies 2D UNET model to the input and returns segmentation & scores
    
    
    Parameters
    ----------
    input_img : 2D numpy array (uint8)
        Input image 
        
    img_pixelsize_x : float
        Input image pixelsize in x direction (pixel width)
        
    img_pixelsize_y : float
        Input image pixelsize in y direction (pixel height)
        
    modelfile_path : string
        Absolute path to HDF5 unet modelfile
        
    weightfile_path : string
        Absolute path to matching HDF5 unet weightfile
        
    iofile_path : string
        Absolute path to location where the temporary caffe input/outputfile 
        will be created. File must not exist.
        
    tiling_x : int
        UNET tile size in X direction 
        
    tiling_y : int
        UNET tile size in Y direction 
        
    gpu_flat : string
        GPU mode. Valid strings are
        '' for CPU mode (default)
        'all' for using all available GPUs
        e.g. '0' for using GPU node 0
        
    cleanup : bool
        If true (default), IO file is deleted after function call.
        
    Returns
    ---------    
    output : dict
        with keys preprocessed_img, scores and segmentation_mask.
        
    '''
    
    #fix parameters
    n_inputchannels=1
    n_iterations=0
    
    def rescale(size,img,mode='uint8'):
        '''Rescales image to new size, using bilinear interpolation.
        
        
        Parameters
        ----------        
        size : tuple
            The new image size in pixels, as a 2-tuple: (width, height)
        
        mode : string
            Datatype to which image is converted before interpolation. Valid strings: ['uint8','float32']'''
        #resize with bilinear interpolation
        
        if mode == 'float32':
            #for floating point images:
            img = np.float32(img)
            img_PIL = PIL.Image.fromarray(img,mode='F')
        elif mode == 'uint8':
            #otherwise:
            img_PIL = PIL.Image.fromarray(img)
        else:
            raise(Exception('Invalid rescaling mode. Use uint8 or float32'))
            
        return np.array(img_PIL.resize(size,PIL.Image.BILINEAR))
    
    def normalize(img):
        ''' MIN/MAX-normalizes image to [0,1] range.'''
        ###normalize image
        img_min = np.min(img)
        img_max = np.max(img)
        img_centered = img - img_min
        img_range = img_max - img_min
        return img_centered / img_range
    
    
    ### prepare image rescaling

    #get model resolution (element size) from modelfile
    modelfile_h5 = h5py.File(modelfile_path,'r')
    modelresolution_y = modelfile_h5['unet_param/element_size_um'][0]
    modelresolution_x = modelfile_h5['unet_param/element_size_um'][1]
    modelfile_h5.close()       
    #get input image absolute size
    abs_size_x = input_img.shape[1] * img_pixelsize_x
    abs_size_y = input_img.shape[0] * img_pixelsize_y
    #get rescaled image size in pixel
    rescaled_size_px_x = int(np.round(abs_size_x / modelresolution_x))
    rescaled_size_px_y = int(np.round(abs_size_y / modelresolution_y))
    rescale_size = (rescaled_size_px_x,rescaled_size_px_y)
    
    ### preprocess image and store in IO file
    
    #normalize image, then rescale
    normalized_img = normalize(input_img)
    rescaled_img = np.float32(rescale(rescale_size,normalized_img,mode='float32'))
    #prepending singleton dimensions to get the desired blob structure
    h5ready_img = rescaled_img[np.newaxis,np.newaxis,:,:]
    #create caffe IO file
    iofile_h5 = h5py.File(iofile_path,mode='x')
    #save image blob to hdf5 dataset "/data"
    iofile_h5.create_dataset('data',data=h5ready_img)
    iofile_h5.close()
    
        
    ### run caffe_unet commands
    
    #assemble sanity check command
    command_sanitycheck = []
    command_sanitycheck.append("caffe_unet")
    command_sanitycheck.append("check_model_and_weights_h5")
    command_sanitycheck.append("-model")
    command_sanitycheck.append(modelfile_path)
    command_sanitycheck.append("-weights")
    command_sanitycheck.append(weightfile_path)
    command_sanitycheck.append("-n_channels")
    command_sanitycheck.append(str(n_inputchannels))
    if gpu_flag:
        command_sanitycheck.append("-gpu")
        command_sanitycheck.append(gpu_flag)
    #runs command and puts console output to stdout
    sanitycheck_proc = subprocess.run(command_sanitycheck,stdout=subprocess.PIPE)
    #aborts if process failed
    sanitycheck_proc.check_returncode()
    
    #assemble prediction command
    command_predict = []
    command_predict.append("caffe_unet")
    command_predict.append("tiled_predict")
    command_predict.append("-infileH5")
    command_predict.append(iofile_path)
    command_predict.append("-outfileH5")
    command_predict.append(iofile_path)
    command_predict.append("-model")
    command_predict.append(modelfile_path)
    command_predict.append("-weights")
    command_predict.append(weightfile_path)
    command_predict.append("-iterations")
    command_predict.append(str(n_iterations))
    command_predict.append("-tile_size")
    command_predict.append(str(tiling_x)+'x'+str(tiling_y))
    command_predict.append("-gpu")
    command_predict.append(gpu_flag)
    if gpu_flag:
        command_predict.append("-gpu")
        command_predict.append(gpu_flag)
    #runs command 
    with subprocess.Popen(command_predict, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            #prints console output
            print(line, end='') 
            
    
    ### read from IO file and postprocess 
    
    # load results from io file and return
    output_h5 = h5py.File(iofile_path)
    score = output_h5['score'].value
    output_h5.close()
    #get segmentation mask by taking channel argmax
    segmentation_mask = np.argmax(score,axis=1)
    
    #cleanup if requested
    if cleanup:
        os.remove(iofile_path)
    
    return dict(preprocessed_img=np.squeeze(h5ready_img),
                scores = np.squeeze(score),
                segmentation_mask = np.squeeze(segmentation_mask))