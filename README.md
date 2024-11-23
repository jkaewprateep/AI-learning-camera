# AI-learning-camera
AI learning image VDO, possible object variables

🧸💬 By using the AI machine learning library TensorFlow allowed the creation of simple iterations method adaptation in Python programming language to train input values from VDO and legend boundary image's object location identification information from learning feedback results of AI machine learning. The program is designed to concatenate the random position of the starting points on XY coordinate and boundary box size for AI machine learning train inputs likelihood of the object in the image inside the boundary as [ 0, 1 ] domain prediction results. </br>

<p align="center" width="100%">
    <img width="55%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/tensorflow_01.png">
    <img width="13.5%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/kid_25.jpg">
    <img width="24.5%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/ratta.png"> </br>
    <b> Learning TensorFlow SDK and Python codes </b> </br>
</p>
</br>
</br>

<p align="center" width="100%">
    <img width="14%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/Movement_detection.gif">
    <img width="11%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/WaterWorld_GamePlay.gif">
    <img width="22%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/girl_drawing.gif"> </br>
    <b> Learning TensorFlow SDK and Python codes </b> </br>
</p>
</br>
</br>

[TF_filters_locations]( https://github.com/jkaewprateep/TF_filters_locations-3/blob/main/README.md ) </br>
[blur_contrast_normalize_filter]( https://github.com/jkaewprateep/blur_contrast_normalize_filter/blob/main/README.md ) </br>

## References and Variables ##

🧸💬 The fast way for directional device communication is to use their drivers or generics communication when APIs can perform input/output as continuous arrays ```cv2```. Examples of using driver APIs include Windows generics drivers, browsers, and client-based APIs ( TAPI, device generic, COM API, and Active-X ).

```
"""""""""""""""""""""""""""""""""""""""""""""
: Library and variables
"""""""""""""""""""""""""""""""""""""""""""""
## Import libraries
import os;
from os.path import exists;

import cv2;                                 # 🧸💬 CV2 for generics API
import tensorflow as tf;                    # 🧸💬 Tensorflow for machine learning
import matplotlib.pyplot as plt;            # 🧸💬 For image plotting
import matplotlib.animation as animation;   # 🧸💬 For animation image plotting

from skimage import measure;                # 🧸💬 For shadow legend plotting

## Variables
frames = []                                 # 🧸💬 Array of VDO frames from generic driver

path = "C:\\Users\\hp\\Videos\\1732345843937.publer.io.mp4"  # 🧸💬 input
cap = cv2.VideoCapture(path)                                 # 🧸💬 Object variable
ret = True                                                   # 🧸💬 Boolean indicates output types
fig, ax = plt.subplots()                                     # 🧸💬 Define subplot for image plotting
filename = "C:\\Users\\hp\\Pictures\\cats\\cat.jpg"          # 🧸💬 Image for loading picture
image = tf.io.read_file( filename, name="image" )            # 🧸💬 Read image file from I/O
image = tf.io.decode_jpeg( image, channels=0 )               # 🧸💬 Decoder as .jpeg image type
image = tf.image.resize( image, [48, 86] )                   # 🧸💬 Image resizing
im = plt.imshow( image )                                     # 🧸💬 Display image background
```

## Callback class ##

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)

		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)
```

## Draw boundary image ##

```
"""""""""""""""""""""""""""""""""""""""""""""
: Image Boundary
"""""""""""""""""""""""""""""""""""""""""""""
def draw_rectang( image, dimensions ) :

    boxes = tf.reshape(dimensions, [1, 1, 4]);
    image = tf.cast(image, dtype=tf.float32);
    
    colors = tf.constant([[255.0, 0.0, 0.0, 0.0]], shape=(1, 4));
    boxes_custom = tf.constant( dimensions, shape=(1, 1, 4)).numpy();
    new_image = tf.image.draw_bounding_boxes( tf.expand_dims(image, axis=0), boxes_custom, colors );
    
    return new_image;
```

## Draw image countour ##

```
"""""""""""""""""""""""""""""""""""""""""""""
: Image legend locator
"""""""""""""""""""""""""""""""""""""""""""""
def find_image_countour( image ):

    image = tf.keras.utils.img_to_array( image )
    image = tf.squeeze( image ).numpy()

    contours = measure.find_contours( image, 200 );

    return contours
```

## Filter ##

```
"""""""""""""""""""""""""""""""""""""""""""""
: Filter
"""""""""""""""""""""""""""""""""""""""""""""
def filters( image ):

    ...
   🧸💬 You can apply filters from CV2, and published filter matrixes example smile faces, edges detection,
        Gaussian-blur and more, but we focus on AI learning variables in this experiment.

    ...
   🧸💬 Variables in the scopes of our experiment are 1. The size is width x height, 2. average of colours
        Pixels ( R, G, B image system ), and position of the image min_x and min_y.
    
    return image
```

## Animate updates ##

```
"""""""""""""""""""""""""""""""""""""""""""""
: Animate
"""""""""""""""""""""""""""""""""""""""""""""
def update( frame ):
    global model;
    global history;
    global iCount;
    global checkpoint_path;
    global dimensions;
    
    iCount = iCount + 1;
    ret, frame = cap.read();
    
    if ( ret ):        
        img = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )
        
        frames.append(img)
        
        ## image
        image = tf.keras.utils.array_to_img(img);
        o_image = tf.image.resize( image, [48, 86] )
        o_image = tf.cast( o_image, dtype=tf.int32 )       
        
        image_forcontour = filters( o_image )
        
        fig.axes[0].clear()
        plt.axis( 'off' )
        
        coords = find_image_countour( image_forcontour )
        
        (width, height) = (48, 86);
        min_x = min([ x for x, y in coords]) // 1;
        max_x = max([ x for x, y in coords]) // 1;
        min_y = min([ y for x, y in coords]) // 1;
        max_y = max([ y for x, y in coords]) // 1;
        
        ###       
        if iCount < 3 :
            dimensions = [ float( min_x / width ), float( min_y / height ), float( max_x / width ),
				float( max_y / height ) ];
        else :
            new_image_data = tf.reshape( o_image, [ 1, 1, 1, 48 * 86 * 3 ] );  
            new_image_data = tf.cast( new_image_data, dtype=tf.float32 );
            size_dimension = tf.constant([min_x, min_y], shape=(1, 1, 1, 2), dtype=tf.float32);
            new_image_data = tf.concat([new_image_data, size_dimension], axis=3);  
        
            result = predict_action( new_image_data );
            print("result: ", result);
            if result == 1 :
                dimensions = [ float( min_x / width ), float( min_y / height ), float( max_x / width ),
				float( max_y / height ) ];
        
        image = draw_rectang( image, dimensions );
        ###
        
        ###
        data_row = create_dataset( o_image, min_x, min_y );
        history = model.fit(data_row, epochs=1, callbacks=[custom_callback]);
        if iCount % 3 == 0 :
            model.save_weights(checkpoint_path);
            
        print( data_row );

        fig.axes[0].axis( 'off' );
        fig.axes[0].grid( False );
        image = tf.squeeze(image);
        image = tf.keras.utils.array_to_img( image );
        im.set_array( image );
        plt.imshow( image );
        
    return im
```

## Prediction ##

```
def predict_action( dataset ):
    global model;

    # print( "dataset: ", dataset.shape );

    predictions = model.predict(tf.expand_dims(tf.expand_dims(tf.squeeze(dataset), axis=0 ), axis=0 ));
    score = tf.nn.softmax(predictions[0])

    return int(tf.math.argmax(score))
```

## Create dataset ##

```
def create_dataset( image_data, init_x, init_y ):
    global listof_items;
    global iCount;
    
    # 🧸💬 Variables in the scopes of our experiment are 1. The size is width x height, 2. average of colours
        # Pixels ( R, G, B image system ), and position of the image min_x and min_y.
        
    # image_data size (48, 86)
    image_data = tf.reshape( image_data, [ 1, 1, 1, 48 * 86 * 3 ] );  
    image_data = tf.cast( image_data, dtype=tf.float32 );
    size_dimension = tf.constant([init_x, init_y], shape=(1, 1, 1, 2), dtype=tf.float32);
    image_data = tf.concat([image_data, size_dimension], axis=3);      
    
    print( image_data.shape );
    
    
    ### LABEL
    ### 1. Result from similarity
    LABEL = tf.constant([0], shape=(1, 1), dtype=tf.float32);
    
    if [init_x, init_y] not in listof_items :
        listof_items.append([init_x, init_y]);
    else :
        LABEL = tf.constant([1], shape=(1, 1), dtype=tf.float32);
    
    ### 2. Result from prediction
    result = predict_action( image_data );
    if iCount % 5 == 0 :
        listof_items.append([init_x, init_y]);
    elif iCount % 6 == 0 :
        listof_items.append([init_x, init_y]);
        
    ### reduce number of focus
    listof_items = listof_items[:-16];
    ###
    
    dataset = tf.data.Dataset.from_tensor_slices((image_data, LABEL));

    return dataset;
```

## Task executor ##

```
"""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""
while(True):
    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
    plt.show()
```
