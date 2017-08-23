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
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QLabel, QFileDialog, QSlider
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt, QTimer

import scipy.misc #library for resizing buffer image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import mv as mv

from PIL import ImageQt

from datetime import datetime

from functools import partial


##################################################  QT DISPLAY ##################################################
class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'Winding Control'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.save_im = False

        self.initGUI()
 
    def initGUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Image viewing region
        self.lbl = QLabel(self)

        # Slider for camera settings on a grid layout (4x4)
        self.exposure = self.create_slider("Exposure", 10, 1e6, 1e5)
        self.gain = self.create_slider("Gain", 0, 18, 10)
        self.blacklevel = self.create_slider("Blacklevel", -100, 100, 0)
        self.framerate = self.create_slider("Framerate", 1, 12, 8)

        grid = QGridLayout()
        grid.addWidget(self.exposure, 0,0)
        grid.addWidget(self.gain, 0,1)
        grid.addWidget(self.blacklevel, 1,0)
        grid.addWidget(self.framerate, 1,1)

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.create_button("Load Image", self.load_image_but))
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

        vbox.setGeometry(10, 10, 200, 30)
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
            print('Changed ' + label + ' to {}'.format(val))

        if label == "Gain":
            val = self.gain.findChild(QSlider).value() 
            print('Changed ' + label + ' to {}'.format(val))

        if label == "Blacklevel":
            val = self.blacklevel.findChild(QSlider).value() 
            print('Changed ' + label + ' to {}'.format(val))

        if label == "Framerate":
            val = self.framerate.findChild(QSlider).value() 
            print('Changed ' + label + ' to {}'.format(val))

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


    def quit(self):
        self.close()


########################## MAIN FUNCTION ##########################

def main():


    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
