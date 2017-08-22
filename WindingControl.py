#                                           #
#                                           #  
#       Created by Janine Müller            #
#                                           #
#                                           #
#       21.08.2017 at TU Dortmund           #
#                                           #
#                                           #
#                                           #
# Loads an selected image into application   #

import sys
import os

from threading import Thread
from six.moves.queue import Queue, Empty, Full

# Qt stuff
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QLabel, QFileDialog
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


#model = load_model('../TrainedModels/Good/811_1611_1611_32_32_int.h5') #loading trained NN


##################################################  Image Aquisition ##################################################

# Class to aquire images from camera
# from https://github.com/geggo/MVacquire/
class AcquisitionThread(Thread):
    
    def __init__(self, device, queue):
        super(AcquisitionThread, self).__init__()
        self.dev = device
        self.queue = queue
        self.wants_abort = False

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

        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Button that allows loading of images
        self.load_button = QPushButton("Load image")
        self.load_button.clicked.connect(self.load_image_but)

        # Button that allows loading of images
        self.save_button = QPushButton("Save image")
        self.save_button.clicked.connect(self.save_image)

        # Button that starts image aquisition
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_aquisition)

        # Button that quits the app
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit)

        # Image viewing region
        self.lbl = QLabel(self)

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.load_button)
        layout_button.addWidget(self.save_button)
        layout_button.addWidget(self.run_button)
        layout_button.addWidget(self.quit_button)
        layout_button.addStretch()

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)
        layout.addWidget(self.lbl)

        self.setLayout(layout)
        self.show()


    @pyqtSlot()
    def run_aquisition(self):
        # Start the thread’s activity.
        # It must be called at most once per thread object. 
        # It arranges for the object’s run() method to be invoked in a separate thread of control.
        # This method will raise a RuntimeError if called more than once on the same thread object.
        self.thread.wants_abort = False
        self.thread.start()

        timer = QTimer(self)
        timer.timeout.connect(self.open)
        timer.start(20) #30 Hz

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
        self.stop_aquisition()
        self.close()


########################## MAIN FUNCTION ##########################

def main():

    #find and open device
    serials = mv.List(0).Devices.children #hack to get list of available device names
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
