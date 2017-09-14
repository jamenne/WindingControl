#############################################################################################
#########                                                                           #########
#########                   Created by Janine Müller                                #########
#########                                                                           #########
#########                                                                           #########
#########                  14.09.2017 at TU Dortmund                                #########
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
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QLabel, QFileDialog, QSlider, QFrame
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg

from keras.models import load_model
import scipy.misc #library for resizing buffer image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mv as mv

from PIL import ImageQt

from datetime import datetime

from functools import partial

import time


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
        self.liveclassification = LiveClassification()
        self.classification = False
        self.save_feed = False
        self.save_img = False


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

            ### Classification of image
            if self.classification == True:
                self.liveclassification.classify_image(imgdata)

            ### Saving Image
            if self.save_img == True:
                self.liveclassification.save_image(imgdata, timestamp)
                self.save_img = False
    
            ## Saving Feed
            if self.save_feed == True:
                img = scipy.misc.imresize(imgdata, (75, 100))
                self.liveclassification.save_image(img, timestamp)

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


    def switch_output(self):
    # ------------------------------------------
    #   This device has 6 DigitalIOs
    # ------------------------------------------
    # IO 0:   Type: Output    Current state: OFF
    # IO 1:   Type: Output    Current state: OFF
    # IO 2:   Type: Output    Current state: OFF
    # IO 3:   Type: Output    Current state: OFF
    # IO 4:   Type: Input     Current state: OFF
    # IO 5:   Type: Input     Current state: OFF
    # ------------------------------------------
    # status = 1: switching GPO ON
    # status = 0: switching GPO OFF
        if self.GPO_status == 1:
            self.dev.Setting.Base.Camera.GenICam.DigitalIOControl.LineInverter=1
        if self.GPOstatus == 0:
            self.dev.Setting.Base.Camera.GenICam.DigitalIOControl.LineInverter=0

############################################  CLASSIFICATION / CAMERA  ##########################################
class LiveClassification:
    def __init__(self):
        super(LiveClassification, self).__init__()

        self.model = load_model('/home/windingcontrol/src/WindingControl/TrainedModels/2017-09-11/167_165_163_200_selu_100epochs.h5')
        self.prob_total = []
        self.save_prob = False
        self.negative = False
        self.means = np.loadtxt('../Means.txt')
        self.stds = np.loadtxt('../StdDev.txt')

        self.path_dir = '/home/windingcontrol/WindingImages/' + str(datetime.now().strftime('%Y-%m-%d') + '/')
        if not os.path.exists(self.path_dir):
            os.makedirs(self.path_dir)
            print('Created path: {}'.format(self.path_dir))

        self.path_dir_Data = '/home/windingcontrol/src/WindingControl/Data/' + str(datetime.now().strftime('%Y-%m-%d') + '/')
        if not os.path.exists(self.path_dir_Data):
            os.makedirs(self.path_dir_Data)
            print('Created path: {}'.format(self.path_dir_Data))



    def classify_image(self,imgdata):

        #resizing image to parse through NN
        img = scipy.misc.imresize(imgdata, (75, 100)) 
        # apply normalization
        img = (img-self.means)/self.stds
        #reshaping data ato parse into keras prediction
        img = np.reshape(img,[1,75,100,1]) 
        #find prediction probability
        Prob = self.model.predict_proba(img, verbose=0) 
        self.prob = np.squeeze(Prob)
        print('{:}'.format(self.prob) )

        if self.save_prob == True:
            self.prob_total.append(self.prob)
        
        if Prob < 0.5:
            print('NEGATIVE')
            self.negative = True
        else:
            print('POSITIVE')
            self.negative = False

    def save_probs_to_file(self):
        
        self.prob_total = np.array(self.prob_total)
        np.savetxt('{}WindingProb_{}.txt'.format(self.path_dir_Data, str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))), self.prob_total)
        print('File written!')
        self.written = True
        self.prob_total = []

    def save_image(self, imgdata, timestamp):

        scipy.misc.toimage(imgdata).save( '{}IMG_{}.jpg'.format(self.path_dir, timestamp) )
        #print('Successfully saved image to file: {:}'.format(self.path_dir) )


    def load_kerasmodel(self, fname):

        self.model = load_model(fname) #loading trained NN
        print('Load selected model')

##################################################  QT DISPLAY ##################################################
class App(QWidget):
 
    def __init__(self, ac_thread, parent=None):
        super(App, self).__init__(parent)
        self.title = 'Winding Control'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.thread = ac_thread
        self.save_im = False
        self.debugtool = None

        self.initGUI()
 
    def initGUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Image viewing region
        self.lbl = QLabel(self)
        self.lbl.setStyleSheet("border: 15px solid white; background-color: white; inset grey")
        self.lbl.setFrameShape(QFrame.StyledPanel)
        self.lbl.setFrameShadow(QFrame.Sunken)

        # Slider for camera settings on a grid layout (4x4)
        self.exposure = self.create_slider("Exposure", 10, 1e6, 58300)
        self.gain = self.create_slider("Gain", 5, 18, 5)
        self.blacklevel = self.create_slider("Blacklevel", -100, 100, 0)
        self.framerate = self.create_slider("Framerate", 1, 20, 15)

        grid = QGridLayout()
        grid.addWidget(self.exposure, 0,0)
        grid.addWidget(self.gain, 0,1)
        grid.addWidget(self.blacklevel, 1,0)
        grid.addWidget(self.framerate, 1,1)

        # RadioButton for saving the Probabilities
        self.check_but1 = QCheckBox('Save_Prob')
        self.check_but1.setChecked(False)
        self.check_but1.stateChanged.connect(partial( self.save_probabilities, self.check_but1 ))

        # RadioButton for saving the video feed
        self.check_but2 = QCheckBox('Save_Feed')
        self.check_but2.setChecked(False)
        self.check_but2.stateChanged.connect(partial( self.save_feed, self.check_but2 ))

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.create_button("Load Model", self.load_kerasmodel_but))
        layout_button.addWidget(self.create_button("Save Image", self.save_image))
        layout_button.addWidget(self.create_button("Show", self.run_aquisition))
        layout_button.addWidget(self.create_button("Start", self.start_classification))
        layout_button.addWidget(self.create_button("Stop", self.stop_classification))
        layout_button.addWidget(self.create_button("Debug", self.start_debugtool))
        layout_button.addWidget(self.create_button("Quit", self.quit))
        layout_button.addWidget(self.check_but1)
        layout_button.addWidget(self.check_but2)
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
        slider.setTickInterval( (maxV-minV+1)/10 )
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
    def load_kerasmodel_but(self):
        """
        Open a File dialog when the button is pressed
        :return:
        """
        
        #Get the file location
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Open')
        # Load the image from the location
        self.thread.liveclassification.load_kerasmodel(self.fname)

    def start_debugtool(self):
        self.debugtool = DebuggingWindow(self.thread)
        self.debugtool.show()

    def save_probabilities(self, b):
        if b.isChecked() == True:
            self.thread.liveclassification.save_prob = True
            print('Probabilities are being saved in array!')

        else:
            if self.thread.liveclassification.save_prob == True:
                self.thread.liveclassification.save_prob = False

                # save collected probs in File
                self.thread.liveclassification.save_probs_to_file()

    def save_feed(self, b):
        if b.isChecked() == True:
            self.thread.save_feed = True
            print('Images are being saved to hard disk!')

        else:
            if self.thread.save_feed == True:
                self.thread.save_feed = False


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
            self.thread.dev.Setting.Base.Camera.GenICam.AcquisitionControl.mvAcquisitionFrameRateEnable=1
            #print('{:d}'.format(self.thread.dev.Setting.Base.Camera.GenICam.AcquisitionControl.mvAcquisitionFrameRateEnable))
            self.thread.dev.Setting.Base.Camera.GenICam.AcquisitionControl.AcquisitionFrameRate=val
            #print('{:.2f}'.format(self.thread.dev.Setting.Base.Camera.GenICam.AcquisitionControl.AcquisitionFrameRate))
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

    def start_classification(self):
        if self.thread.liveclassification is None:
            self.thread.liveclassification = LiveClassification()
        
        if self.thread.liveclassification.model is not None:
            print('Start Classification')
            self.thread.liveclassification.prob_total = []
            self.thread.classification = True
        else:
            print('Load Model first!')


    def stop_classification(self):
        print('Stop Classification')
        self.thread.classification = False
        self.thread.plotting = False
        if self.thread.liveclassification.save_prob == True:
            # save collected probs in File
            self.thread.liveclassification.save_probs_to_file()

    def open(self):
        try:
            img = self.thread.queue.get(block=True, timeout = 1)
            q = QPixmap.fromImage(ImageQt.ImageQt(scipy.misc.toimage(img['img'])))

            if self.thread.liveclassification.negative == True:
                self.lbl.setStyleSheet("border: 15px solid red")

            if self.thread.liveclassification.negative == False:
                self.lbl.setStyleSheet("border: 15px solid green")

            self.lbl.setPixmap(q)
            self.lbl.adjustSize()
            self.show()

        except Empty:
            print("got no image")

    def save_image(self):
        self.thread.save_img = True

    def quit(self):
        if self.thread.running == True:
            self.stop_aquisition()
        if self.debugtool is not None:
            self.debugtool.quit()
        self.close()


##########################   Debugging   ##########################

class DebuggingWindow(QWidget):

    def __init__(self, ac_thread, parent=None):
        super(DebuggingWindow, self).__init__(parent)
        self.title = "Debugging Tool"
        self.thread = ac_thread
        self.t = QTime()
        self.timer = QTimer()
        self.x = np.arange(50)
        self.y = np.ones(50)
        self.all_y = []

        ## Switch to using white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')


        self.initDebug()

    def initDebug(self):

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        #layout_button.addWidget(self.create_button("Plotting", self.start_live_plotting))
        layout_button.addWidget(self.create_button("Quit", self.quit))
        layout_button.addWidget(self.create_button("Plot", self.plot))
        layout_button.addWidget(self.create_button("Stop", self.stop_plotting))
        layout_button.addStretch()

        self.plotWidget1 = pg.PlotWidget()
        self.plotWidget2 = pg.PlotWidget()
        self.timer.timeout.connect(self._update)

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)
        layout.addWidget(self.plotWidget1)
        layout.addWidget(self.plotWidget2)

        self.setLayout(layout)

        self.t.start()

    def create_button(self, label, func):

        button = QPushButton(label)
        button.clicked.connect(func)

        return button


    @pyqtSlot()
    def stop_plotting(self):
        self.timer.stop()

    def plot(self):

        self.timer.start(200)
        self.plotWidget1.clear()       

        self.curve1 = self.plotWidget1.plot(self.x, self.y, pen=4, symbol='o') 

        self.plotWidget2.clear()
        ## compute standard histogram
        e1, e2 = np.histogram(self.y, bins=np.linspace(-3, 8, 40))

        self.curve2 = self.plotWidget2.plot(e2, e1, stepMode=True, fillLevel=0, brush=(0,0,255,150))

    def _update(self):
        if self.thread.classification == True:
            self.x = np.roll(self.x, -1)
            self.y = np.roll(self.y, -1)
            self.x[-1] = self.t.elapsed()
            self.y[-1] = self.thread.liveclassification.prob

            self.all_y.append(self.thread.liveclassification.prob)

            self.curve1.setData(x=self.x, y=self.y)

            ## compute standard histogram
            e1, e2 = np.histogram(self.all_y, bins=np.linspace(-1, 1, 10))
            self.curve2.setData(x=e2, y=e1)


        
    def quit(self):
        self.close()


########################## MAIN FUNCTION ##########################

def main():
    #device.Setting.Base.Camera.GenICam.DigitalIOControl.LineInverter=1 to set output ON

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
