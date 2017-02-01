#Behavioral Cloning

This neural network was developed with the goal of steering a simulated car around a track with no input from the user. The project was completed as part of Udacity's Self-Driving Car Nano Degree. The network was heavily based on the architecture described in Nvidia's "End to End Learning for Self-Driving Cars" (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). A few tweaks were made to the architecture in order to streamline the preprocessing of images fed into the network. The Nvidia model is a good basis for this project since it is relatively simple and has been proven to have good real world performance in completing a very similar task.

The network uses images captured from a virtual, front facing camera, centered on the car as input and the corresponding instantaneous steering angle as a label for the data. The base dataset was supplied by Udacity and additional input data was gathered using a training module within the simulator software. The additional data was created in order to give the network examples of the correct response in areas where the car was leaving the track during testing.

##Network Architecture

The network uses the following structure:
1.  Cropping layer
  * Removes pixel data from the top of each image. The area of the image removed contains only the sky and is not helpful for training the model to steer the car.
2.  Resize layer
  * The Nvidia architecture is designed for a 66x200 pixel input image - this layer scales the image to the expected size.
3.  Normalization layer
  * Image data is normalized to values between -0.5 and 0.5. This generally yields better results in training.
