#############################################################################################
#########                                                                           #########
#########                   Created by Janine Müller                                #########
#########                                                                           #########
#########                                                                           #########
#########                  21.08.2017 at TU Dortmund                                #########
#########                                                                           #########
#########                                                                           #########
#########                                                                           #########
#########                                                                           #########
#########       'Run' displays a camera feed with the mvBlueFox3 using PyQt         #########
#########       Loads a selected images with the 'Load' button                      #########
#########       Saves a displayed image with the 'Save' button to hard drive        #########
#########       'Quit' determines the application                                   #########
#########                                                                           #########
#############################################################################################


import sys
import os

from threading import Thread
from six.moves.queue import Queue, Empty, Full

# Qt stuff
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QLabel, QFileDialog, QSlider, QFrame
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt, QTimer

from keras.models import load_model
import scipy.misc #library for resizing buffer image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mv as mv

from PIL import ImageQt

from datetime import datetime

from functools import partial


##################################################  Image Aquisition ##################################################

# Class to aquire images from camera
# from https://github.com/geggo/MVacquire/
class AcquisitionThread(Thread):
    
    def __init__(self, device, queue):
        super(AcquisitionThread, self).__init__()
        self.dev = device
        self.queue = queue
        self.wants_abort = False
        self.running = False
        self.classification = False

    def acquire_image(self):
        #try to submit 2 new requests -> queue always full
        try:
            self.dev.image_request()
            self.dev.image_request()
        except mv.MVError as e:
            pass

        #get image
        image_result = None
        try:
            image_result = self.dev.get_image()
        except mv.MVTimeoutError:
            print("timeout")
        except Exception as e:
            print("camera error: ",e)
        
        #pack image data together with metadata in a dict
        if image_result is not None:
            buf = image_result.get_buffer()
            imgdata = np.array(buf, copy = False)

            ### Classification of image

            if self.classification == True:
                img = scipy.misc.imresize(imgdata, (75, 100)) #resizing image to parse through NN
                img = np.reshape(img,[1,75,100,1]) #reshaping data ato parse into keras prediction
                ClassProb = self.model.predict_proba(img, verbose=0) #find prediction probability
                print(ClassProb)
                
                if ClassProb[0,0] < 0.5:
                    print('NEGATIVE')
                    #print("\a")
                else:
                    print('POSITIVE')

            ### End of Classification
            
            info=image_result.info
            timestamp = info['timeStamp_us']
            frameNr = info['frameNr']

            del image_result
            return dict(img=imgdata, t=timestamp, N=frameNr)
        
    def reset(self):
        self.dev.image_request_reset()
    
    # Method representing the thread’s activity.
    # You may override this method in a subclass. - YES, we'll do it HERE -
    # The standard run() method invokes the callable object passed to the object’s constructor as the target argument, 
    # if any, with sequential and keyword arguments taken from the args and kwargs arguments, respectively.    
    def run(self):
        self.reset()
        while not self.wants_abort:
            img = self.acquire_image()
            if img is not None:
                try:
                    self.queue.put_nowait(img)

                    #print('.',) #
                except Full:
                    #print('!',)
                    pass

        self.reset()
        print("acquisition thread finished")

    def stop(self):
        self.wants_abort = True


##################################################  QT DISPLAY ##################################################
class App(QWidget):
 
    def __init__(self, ac_thread):
        super().__init__()
        self.title = 'Winding Control'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.thread = ac_thread
        self.save_im = False

        self.initGUI()
 
    def initGUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Image viewing region
        self.lbl = QLabel(self)
        self.lbl.setStyleSheet("border: 5px solid green; background-color: white; inset grey")
        self.lbl.setFrameShape(QFrame.StyledPanel)
        self.lbl.setFrameShadow(QFrame.Sunken)

        # Slider for camera settings on a grid layout (4x4)
        self.exposure = self.create_slider("Exposure", 10, 1e6, 130000)
        self.gain = self.create_slider("Gain", 0, 18, 1)
        self.blacklevel = self.create_slider("Blacklevel", -100, 100, 0)
        self.framerate = self.create_slider("Framerate", 1, 12, 8)

        grid = QGridLayout()
        grid.addWidget(self.exposure, 0,0)
        grid.addWidget(self.gain, 0,1)
        grid.addWidget(self.blacklevel, 1,0)
        grid.addWidget(self.framerate, 1,1)

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.create_button("Load Model", self.load_kerasmodel_but))
        layout_button.addWidget(self.create_button("Save Image", self.save_image))
        layout_button.addWidget(self.create_button("Run", self.run_aquisition))
        layout_button.addWidget(self.create_button("Start", self.run_classification))
        layout_button.addWidget(self.create_button("Stop", self.stop_classification))
        layout_button.addWidget(self.create_button("Quit", self.quit))
        layout_button.addStretch()

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)
        layout.addLayout(grid)
        layout.addWidget(self.lbl)

        self.setLayout(layout)
        self.show()

    def create_slider(self, label, minV, maxV, value):
        groupBox = QGroupBox(label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(minV)
        slider.setMaximum(maxV)
        slider.setValue(value)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval( (maxV-minV)/10 )
        slider.setFocusPolicy(Qt.NoFocus)

        slider.valueChanged.connect(partial (self.slider_val_changed,  label) )

        vbox = QVBoxLayout()
        vbox.addWidget(slider)
        vbox.addStretch(2)

        #vbox.setGeometry(10, 10, 200, 30)
        groupBox.setLayout(vbox)

        return groupBox

    def create_button(self, label, func):

        button = QPushButton(label)
        button.clicked.connect(func)

        return button



    @pyqtSlot()
    def slider_val_changed(self, label):

        if label == "Exposure":
            val = self.exposure.findChild(QSlider).value()
            self.thread.dev.Setting.Base.Camera.GenICam.AcquisitionControl.ExposureTime=val
            print('Changed Exposure Time to {}'.format(val))

        if label == "Gain":
            val = self.gain.findChild(QSlider).value()
            self.thread.dev.Setting.Base.Camera.GenICam.AnalogControl.Gain=val
            print('Changed ' + label + ' to {}'.format(val))

        if label == "Blacklevel":
            val = self.blacklevel.findChild(QSlider).value()
            self.thread.dev.Setting.Base.Camera.GenICam.AnalogControl.BlackLevel=val
            print('Changed ' + label + ' to {}'.format(val))

        if label == "Framerate":
            val = self.framerate.findChild(QSlider).value() 
            print('Changed ' + label + ' to {}'.format(val))


    def run_aquisition(self):
        # Start the thread’s activity.
        # It must be called at most once per thread object. 
        # It arranges for the object’s run() method to be invoked in a separate thread of control.
        # This method will raise a RuntimeError if called more than once on the same thread object.
        self.thread.wants_abort = False
        self.thread.start()
        self.thread.running = True

        timer = QTimer(self)
        timer.timeout.connect(self.open)
        timer.start(20) #30 Hz

    def run_classification(self):
        print('Start Classification')
        self.thread.classification = True

    def stop_classification(self):
        print('Stop Classification')
        self.thread.classification = False

    def open(self):
        try:
            img = self.thread.queue.get(block=True, timeout = 1)
            q = QPixmap.fromImage(ImageQt.ImageQt(scipy.misc.toimage(img['img'])))

            if self.save_im:
                path = '/home/windingcontrol/WindingImages/IMG_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.jpg'
                scipy.misc.toimage(img['img']).save(path)
                print('Successfully saved image to file: {:}'.format(path))
                self.save_im = False

            
            self.lbl.setPixmap(q)
            self.lbl.adjustSize()
            self.show()

        except Empty:
            print("got no image")

    def save_image(self):
        self.save_im = True

    def load_kerasmodel_but(self):
        """
        Open a File dialog when the button is pressed
        :return:
        """
        
        #Get the file location
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Open')
        # Load the image from the location
        self.load_kerasmodel()

    def load_kerasmodel(self):

        self.thread.model = load_model(self.fname) #loading trained NN



    def load_image_but(self):
        """
        Open a File dialog when the button is pressed
        :return:
        """
        
        #Get the file location
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Open')
        # Load the image from the location
        self.load_image()

    def load_image(self):
        """
        Set the image to the pixmap
        :return:
        """
        pixmap = QPixmap(self.fname)
        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
        self.lbl.setPixmap(pixmap)

    def stop_aquisition(self):
        #wait until acquisition thread has stopped
        self.thread.stop()

        # Wait until the thread terminates. This blocks the calling thread until the thread whose join() method is called terminates 
        # – either normally or through an unhandled exception – or until the optional timeout occurs.
        # When the timeout argument is present and not None, it should be a floating point number 
        # specifying a timeout for the operation in seconds (or fractions thereof). 
        # As join() always returns None, you must call is_alive() after join() to decide whether a timeout happened 
        # – if the thread is still alive, the join() call timed out.
        # When the timeout argument is not present or None, the operation will block until the thread terminates.
        # A thread can be join()ed many times.
        # join() raises a RuntimeError if an attempt is made to join the current thread as that would cause a deadlock. 
        # It is also an error to join() a thread before it has been started and attempts to do so raise the same exception.
        self.thread.join()


    def quit(self):
        if self.thread.running:
            self.stop_aquisition()
        self.close()


########################## MAIN FUNCTION ##########################

def main():

    #find and open device
    serials = mv.List(0).Devices.children
    serial = serials[0]
    device = mv.dmg.get_device(serial)
    print('Using device:', serial)

    queue = Queue(10)
    acquisition_thread = AcquisitionThread(device, queue)

    app = QApplication(sys.argv)
    ex = App(acquisition_thread)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
