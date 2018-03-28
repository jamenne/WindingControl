# Winding Control

Repository contains files for the training and implementation of a Convolutional
Neural Network (CNN) for the quality control of the Scintilating Fibre Mat 
winding process

## HOW TO RUN
Install required python packages:  
* Numpy
* SciPy
* PyQt5
* PIL
* matplotlib
* keras

Install MVacquire python wrapper:  
* https://github.com/geggo/MVacquire

In Directory run the command:  
* python WindingControl.py
* 'Run' displays a camera feed with the mvBlueFox3 using PyQt         
* 'Save' will save the displayed image to hard drive        
* 'Quit' determines the application
* Camera setting handling available
* Classification of incoming images working
* 'Load' will prompt you to select a model for classification and load it
* 'Start' will start the classification of images
* 'Stop' will abort the classification of images
* 'Save_Prob' will save classification probabilities to an array and after Classification is stopped or 'Save_Prob' is unchecked it will be saved to a file


## for image processing
* converts images from jpg to 600x800 bmp
* remove .jpg ending and places .bmp
* for i in *.jpg; do sips -s format bmp -s formatOptions 70 "${i}" -z 600 800 --out "${i%jpg}bmp"; done