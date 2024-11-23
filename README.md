# AI-learning-camera
AI learning image VDO, possible object variables

<p align="center" width="100%">
    <img width="40%" src="https://github.com/jkaewprateep/TF-and-JSON/blob/main/tensorflow_01.png">
    <img width="14%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/Movement_detection.gif">
    <img width="11%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/WaterWorld_GamePlay.gif">
    <img width="22%" src="https://github.com/jkaewprateep/AI-learning-camera/blob/main/girl_drawing.gif"> </br>
    <b> Learning TensorFlow SDK and Python codes </b> </br>
</p>
</br>
</br>

## References and Variables ##

🧸💬 The fast way for directional device communication is to use their drivers or generics communication when APIs can perform input/output as continuous arrays ```cv2```. Examples of using driver APIs include Windows generics drivers, browsers, and client-based APIs ( TAPI, device generic, COM API, and Active-X ).

```
## Import libraries
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

## Draw boundary image ##

```
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
def find_image_countour( image ):

    image = tf.keras.utils.img_to_array( image )
    image = tf.squeeze( image ).numpy()

    contours = measure.find_contours( image, 200 );

    return contours
```

## Animate updates ##

```
def update( frame ):
    global iCount;

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
        
        dimensions = [ float( min_x / width ), float( min_y / height ), float( max_x / width ), float( max_y / height ) ];
        print(dimensions);
        
        image = draw_rectang( image, dimensions );
        ###

        fig.axes[0].axis( 'off' )
        fig.axes[0].grid( False 
        image = tf.squeeze(image);
        image = tf.keras.utils.array_to_img( image );
        im.set_array( image );
        plt.imshow( image );
        
    return im
```
