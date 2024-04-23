# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
# define a video capture object
vid = cv2.VideoCapture(0)
model = tf.keras.models.load_model("keras_model.h5")
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(224,224))
    test_image = np.expand_dims(frame, axis=0)
    test_image = test_image/255.0
    perdiction = model.perdict(test_image)
    print(perdiction[0])
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()