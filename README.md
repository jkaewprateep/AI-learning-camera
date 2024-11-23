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
