from __future__ import print_function

import time
from threading import Thread
from six.moves.queue import Queue, Empty, Full
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox, QLabel
from PyQt5.QtWidgets import QCalendarWidget, QFontDialog, QColorDialog, QTextEdit, QFileDialog, QScrollArea
from PyQt5.QtWidgets import QCheckBox, QProgressBar, QComboBox, QLabel, QStyleFactory, QLineEdit, QInputDialog
from PIL import Image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mv
import keras
import sys
import os
from PIL import ImageQt
import scipy


from keras.models import load_model
import scipy.misc #library for resizing buffer image


model = load_model('811_169_167_32_32_int.h5') #loading trained NN


class AcquisitionThread(Thread):
    
    def __init__(self, device, queue):
        super(AcquisitionThread, self).__init__()
        self.dev = device
        self.queue = queue
        self.wants_abort = False
        #self.hist = np.empty([1,2])

        


    
    def acquire_image(self):
        #try to submit 2 new requests so that queue is always full
        try:
            self.dev.image_request()
            self.dev.image_request()
        #will quit if there is a camera error    
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
            AcquisitionThread.image = np.array(imgdata)
            img = scipy.misc.imresize(imgdata, (75, 100)) #resizing image to parse through NN
            img = np.reshape(img,[1,75,100,1]) #reshaping data ato parse into keras prediction
            ClassProb = model.predict_proba(img, verbose=0) #find prediction probability
            print(ClassProb)
            
            with open("hist.dat", "a") as myfile:
                    myfile.write(ClassProb)
            #self.hist = np.append(self.hist, ClassProb, axis=0)
            #Output to show if image is positive/negative
            #np.savetxt('hist.dat', self.hist, fmt='%.18e', delimiter=' ', newline='\n')
            if ClassProb[0,0] < 0.5:
                print('NEGATIVE')
                #print("\a")
            else:
                print('POSITIVE')
                
            info=image_result.info
            timestamp = info['timeStamp_us']
            frameNr = info['frameNr']

            del image_result
            return dict(img=imgdata, t=timestamp, N=frameNr)
        
    def reset(self):
        self.dev.image_request_reset()
        
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

class  Window(QMainWindow):

    def __init__(self):
    	super(Window, self).__init__()
    	self.setGeometry(0,0,1610,1100)
    	self.setWindowIcon(QIcon('god_particle.jpg'))
    	self.imageLabel = QLabel()
    	self.imageLabel.setScaledContents(True)
    	self.scrollArea = QScrollArea()
    	self.scrollArea.setWidget(self.imageLabel)
    	self.setCentralWidget(self.scrollArea)
    	self.setWindowTitle("Winding Control")
    	    	
    	timer = QTimer(self)
    	timer.timeout.connect(self.open)
    	timer.start(20) #30 Hz


    #def button1(self):
     #   btn = QPushButton("Begin Acquisition", self)
      #  btn.clicked.connect(acquisition_thread.start)
       # btn.move(0,0)
        #btn.resize(btn.sizeHint())


#get data and display
    def open(self):
        img = AcquisitionThread.image
        q = QPixmap.fromImage(ImageQt.ImageQt(scipy.misc.toimage(img)))              
        
        self.imageLabel.setPixmap(q)
        self.imageLabel.adjustSize()
        self.show()

#find an open device
serials = mv.List(0).Devices.children #hack to get list of available device names
serial = serials[0]
device = mv.dmg.get_device(serial)
print('Using device:', serial)

queue = Queue(10)
acquisition_thread = AcquisitionThread(device, queue)
#consume images in main thread
if __name__ == "__main__":  # had to add this otherwise app crashed

    def run():
        app = QApplication(sys.argv)
        Gui = Window()
        sys.exit(app.exec_())


acquisition_thread.start()
run()
while True:
    try:
        
        img = queue.get(block=True, timeout = 1)
        Gui.show()
        char = sys.stdin.read(1)
    except char:
        print("got no image")
      

#wait until acquisition thread has stopped
acquisition_thread.stop()
acquisition_thread.join()

