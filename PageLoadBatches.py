import sys
import os
import numpy as np
import cv2
import glob
import itertools

def getImageArr(path, width, height, imgNorm="divide", ordering='channels_last'):
    try:
        # Try loading as image first
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # If image loading failed, try loading as .npy
        if img is None and path.endswith('.npy'):
            img = np.load(path)
            if img.ndim == 3:  # If RGB, convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Verify we have valid image data
        if img is None or img.size == 0:
            raise ValueError(f"Empty image data from {path}")
            
        # Resize and normalize
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        
        if imgNorm == "divide":
            img /= 255.0
        
        # Add channel dimension
        if ordering == 'channels_last':
            img = np.expand_dims(img, axis=-1)
        else:
            img = np.expand_dims(img, axis=0)
            
        return img
        
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        # Return zero array with correct shape
        if ordering == 'channels_last':
            return np.zeros((height, width, 1))
        else:
            return np.zeros((1, height, width))

def getSegmentationArr(path, nClasses, width, height):
    try:
        # Load mask (grayscale)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None and path.endswith('.npy'):
            mask = np.load(path)
        
        mask = cv2.resize(mask, (width, height))
        
        # For binary segmentation
        if nClasses == 1:
            mask = (mask > 127).astype(np.float32)  # Threshold at 127
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
            return mask
        else:
            # Multi-class handling
            seg_labels = np.zeros((height, width, nClasses))
            for c in range(nClasses):
                seg_labels[:,:,c] = (mask == c).astype(int)
            return seg_labels
            
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return np.zeros((height, width, nClasses if nClasses > 1 else 1))
    

def imageSegmentationGenerator(images_path, seg_path, batch_size, n_classes, 
                             input_height, input_width, output_height, output_width):
    # Support both directory paths and file lists
    if isinstance(images_path, str):
        assert images_path[-1] == '/'
        images = sorted(glob.glob(images_path + "*.jpg") + 
                      glob.glob(images_path + "*.png") + 
                      glob.glob(images_path + "*.npy"))
    else:
        images = sorted(images_path)
        
    if isinstance(seg_path, str):
        assert seg_path[-1] == '/'
        segmentations = sorted(glob.glob(seg_path + "*.jpg") + 
                             glob.glob(seg_path + "*.png") + 
                             glob.glob(seg_path + "*.npy"))
    else:
        segmentations = sorted(seg_path)
    
    # Verify we have pairs
    assert len(images) == len(segmentations), \
           f"Number of images ({len(images)}) and masks ({len(segmentations)}) don't match"
    
    # More flexible filename matching
    for im, seg in zip(images, segmentations):
        im_name = os.path.splitext(os.path.basename(im))[0]
        seg_name = os.path.splitext(os.path.basename(seg))[0]
        
        if not (im_name == seg_name or im_name == seg_name.replace('_mask', '')):
            print(f"Warning: Potential mismatch - Image: {im_name}, Mask: {seg_name}")
    
    zipped = itertools.cycle(zip(images, segmentations))
    
    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))
        
        yield np.array(X), np.array(Y)
        