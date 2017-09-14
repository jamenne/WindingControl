#############################################################################################
#########                                                                           #########
#########                   Created by Janine Müller                                #########
#########                                                                           #########
#########                                                                           #########
#########                  21.08.2017 at TU Dortmund                                #########
#########                                                                           #########
#########                                                                           #########
#########                                                                           #########
#########                  Testing skript, without talking to camera                #########
#########                                                                           #########
#########                                                                           #########
#########                                                                           #########
#########                                                                           #########
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

import scipy.misc

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageQt

from datetime import datetime

from functools import partial

import time


##################################################  Image Aquisition ##################################################

# Class to aquire images from camera
# from https://github.com/geggo/MVacquire/
class AcquisitionThread(Thread):
    
    def __init__(self, queue):
        super(AcquisitionThread, self).__init__()
        self.queue = queue
        self.wants_abort = False
        self.running = False
        self.classification = False
        self.negative = False
        self.save_class = False
        self.written = False
        self.ClassProb_total = []
        self.prob = 0
        self.plotting = False

        self.path_dir_Data = '/home/windingcontrol/src/WindingControl/Data/' + str(datetime.now().strftime('%Y-%m-%d') + '/')
        if not os.path.exists(self.path_dir_Data):
            os.makedirs(self.path_dir_Data)
            print('Created path: {}'.format(self.path_dir_Data))

    def acquire_image(self):
        image_result = 1

        ### Classification of image

        if self.classification == True:
            image_result = np.random.uniform(0,1)
            self.prob = image_result
            print('{:}'.format(self.prob) )

            if self.save_class ==True:
                self.ClassProb_total.append(self.prob)
            
            if self.prob < 0.5:
                print('NEGATIVE')
                self.negative = True
            else:
                print('POSITIVE')
                self.negative = False

        ### End of Classification


        return image_result
    
    # Method representing the thread’s activity.
    # You may override this method in a subclass. - YES, we'll do it HERE -
    # The standard run() method invokes the callable object passed to the object’s constructor as the target argument, 
    # if any, with sequential and keyword arguments taken from the args and kwargs arguments, respectively.    
    def run(self):
        while not self.wants_abort:
            img = self.acquire_image()
            if img is not None:
                try:
                    self.queue.put_nowait(img)

                    #print('.',) #
                except Full:
                    #print('!',)
                    pass

        print("acquisition thread finished")

    def stop(self):
        self.wants_abort = True


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

        # RadioButton for saving the Probabilities
        self.check_but1 = QCheckBox('Save_Prob')
        self.check_but1.setChecked(False)
        self.check_but1.stateChanged.connect(partial( self.save_probabilities, self.check_but1 ))

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.create_button("Run", self.run_aquisition))
        layout_button.addWidget(self.create_button("Start", self.run_classification))
        layout_button.addWidget(self.create_button("Stop", self.stop_classification))
        layout_button.addWidget(self.create_button("Debug", self.start_debugtool))
        layout_button.addWidget(self.create_button("Quit", self.quit))
        layout_button.addWidget(self.check_but1)
        layout_button.addStretch()

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)
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
    def save_probabilities(self, b):
        if b.isChecked() == True:
            self.thread.save_class = True
            print('Probabilties are being saved in array!')
        else:
            if self.thread.save_class == True:
                self.thread.save_class = False
                self.thread.ClassProb_total = np.array(self.thread.ClassProb_total)
                np.savetxt(self.thread.path_dir_Data + "WindingProb_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".txt", self.thread.ClassProb_total)
                print('File written!')
                self.thread.written = True
                self.thread.ClassProb_total = []
                self.thread.save_class = True


    # Button Actions
    def start_debugtool(self):
        self.debugtool = DebuggingWindow(self.thread)
        self.debugtool.show()


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
        self.thread.ClassProb_total = []
        self.thread.classification = True


    def stop_classification(self):
        print('Stop Classification')
        self.thread.classification = False
        self.thread.plotting = False

        if self.thread.save_class == True:
            self.thread.ClassProb_total = np.array(self.thread.ClassProb_total)
            np.savetxt(self.thread.path_dir_Data + "WindingProb_" + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".txt", self.thread.ClassProb_total)
            print('File written!')
            self.thread.written = True

    def open(self):
        try:
            img = self.thread.queue.get(block=True, timeout = 1)
            #rndImage = np.random.rand(600, 800, 3)
            
            #q = QPixmap.fromImage(ImageQt.ImageQt(scipy.misc.toimage(rndImage)))



            if self.thread.negative == True:
                self.lbl.setStyleSheet("border: 15px solid red")

            if self.thread.negative == False:
                self.lbl.setStyleSheet("border: 15px solid green")

            #self.lbl.setPixmap(q)
            self.lbl.adjustSize()
            self.show()

        except Empty:
            print("got no image")

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
        self.thread.plotting = False

    def plot(self):
        self.thread.plotting = True

        self.timer.start(200)
        self.plotWidget1.clear()       

        self.curve1 = self.plotWidget1.plot(self.x, self.y, pen=4, symbol='o') 

        self.plotWidget2.clear()
        ## compute standard histogram
        e1, e2 = np.histogram(self.y, bins=np.linspace(-3, 8, 40))

        self.curve2 = self.plotWidget2.plot(e2, e1, stepMode=True, fillLevel=0, brush=(0,0,255,150))

    def _update(self):
        if self.thread.plotting == True:
            self.x = np.roll(self.x, -1)
            self.y = np.roll(self.y, -1)
            self.x[-1] = self.t.elapsed()
            self.y[-1] = self.thread.prob

            self.all_y.append(self.thread.prob)

            self.curve1.setData(x=self.x, y=self.y)

            ## compute standard histogram
            e1, e2 = np.histogram(self.all_y, bins=np.linspace(-1, 1, 10))
            self.curve2.setData(x=e2, y=e1)
        else: 
            self.timer.stop()

        


    def quit(self):
        self.close()

########################## MAIN FUNCTION ##########################

def main():

    queue = Queue(10)
    acquisition_thread = AcquisitionThread(queue)

    app = QApplication(sys.argv)
    ex = App(acquisition_thread)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
