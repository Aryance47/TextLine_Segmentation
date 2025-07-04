


from keras.models import *
from keras.layers import *



import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/vgg.h5"

IMAGE_ORDERING = 'channels_first' 

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
	o_shape2 = Model( i  , o2 ).output_shape
	outputHeight2 = o_shape2[2]
	outputWidth2 = o_shape2[3]

	o_shape1 = Model( i  , o1 ).output_shape
	outputHeight1 = o_shape1[2]
	outputWidth1 = o_shape1[3]

	cx = abs( outputWidth1 - outputWidth2 )
	cy = abs( outputHeight2 - outputHeight1 )

	if outputWidth1 > outputWidth2:
		o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
	else:
		o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
	
	if outputHeight1 > outputHeight2 :
		o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
	else:
		o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

	return o1 , o2 

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def FCN8(nClasses, input_height, input_width):
    inputs = Input(shape=(input_height, input_width, 1))
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 1/2 size
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 1/4 size
    
    # Bottleneck at 1/4 size
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    
    # Decoder
    up1 = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(conv3)  # 1/2 size
    merge1 = concatenate([conv2, up1])
    
    up2 = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(merge1)  # original size
    merge2 = concatenate([conv1, up2])
    
    # Ensure output matches input spatial dimensions
    outputs = Conv2D(nClasses, 1, activation='sigmoid')(merge2)
    
    return Model(inputs=inputs, outputs=outputs)
