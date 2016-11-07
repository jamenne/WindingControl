// WindingControl.cpp
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//

#include "WindingTest.h"

//OpenCV stuff
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp> // for video stabilisation
#include <opencv2/nonfree/nonfree.hpp>

#include <string>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;


void ChooseRect(int event, int x, int y, int flags, void* output)
{   
    // vector to store the output Points
    vector<Point>* Out = (vector<Point>*) output;
    // needs to be static, otherwise 0 in between mouse actions (you'll lost the first corner) 
    static Point P1;
    static Point P2;
    bool clicked = false;
    if  ( event == EVENT_LBUTTONDOWN )
    {
        // Left button of mouse is clicked
        clicked = true;
        // Points are set
        P1.x = x;
        P1.y = y;
        P2.x = x;
        P2.y = y;

    }
    else if (event == EVENT_LBUTTONUP)
    {
        // Left button of mouse is released
        // second Point is set to final position
        P2.x = x;
        P2.y = y;
        clicked = false;
        // store Points to vector to transfer them outside the function
        Out->push_back(P1);
        Out->push_back(P2);
    }

    else if (event == EVENT_MOUSEMOVE)
    {
        // mouse is moving

        //always store new position to Point P2
        if (clicked)
        {
            P2.x = x;
            P2.y = y;
        }
    }
    //cout << "P1:\t" << P1.x << "\t" << P1.y << "\tP2:\t" << P2.x << "\t" << P2.y << endl;

}

void TestCallBack(){

// Read image from file 
Mat img = imread("/Users/janine/Desktop/2267049_web.jpg");

//if fail to read the image
if ( img.empty() ) 
{ 
    cout << "Error loading the image" << endl;
    exit(EXIT_FAILURE);
}

//Create a window
namedWindow("My Window", 1);

//vector to store data
vector<Point> Out;

//set the callback function for any mouse event
setMouseCallback("My Window", ChooseRect, &Out);

//show the image
imshow("My Window", img);

// Wait until user press some key
waitKey(0);

if (!Out.empty())
{
    cout << Out[0].x << "\t" << Out[0].y << "\t" << Out[1].x << "\t" << Out[1].y << endl;   
}

}

void ProzessCapturedFrame2(string VideoName){


    VideoCapture cap; // open the video file for reading

    namedWindow("Original Video",CV_WINDOW_NORMAL); //create a window called "Original Video"
    resizeWindow("Original Video", 800, 600);

    Mat frame, frame_grey;

    char key('a');
    int wait_ms(10);
    bool stab = false;
    bool tracking = true;
    
    while(key != 27 && key != 'q'){

        cap.open(VideoName);

        if ( !cap.isOpened() ){  // if not success, exit program

         cout << "Cannot open the video file" << endl;
         exit(EXIT_FAILURE);
        }

        //check if the video has reach its last frame
        while(cap.get(CV_CAP_PROP_POS_FRAMES)<cap.get(CV_CAP_PROP_FRAME_COUNT) && key != 27 && key != 'q'){

            if(waitKey(wait_ms) == 116) // t has been pressed
            {
                stab=true;
                cout << "Stabilization enabled" << endl;
            }

            else if (waitKey(wait_ms) == 102)
            {
                stab=false;
                cout << "Stabilization disabled" << endl;
            }


            bool bSuccess = cap.read(frame); // read a new frame from video
            resize( frame, frame, Size(800, 600), 0, 0, INTER_CUBIC) ; //resize the image
            cvtColor(frame, frame_grey, COLOR_BGR2GRAY);
            
            imshow("Original Video", frame_grey); //show the frame in "Original Video" window

            if ( !bSuccess ) //if not success, break loop
            {
                cout << "Cannot read the frame from video file" << endl;
                break;
            }

            // Video stabilisation
            if (stab)
            {
                //frame.convertTo(frame_grey, CV_8U);
                //cout << "Mat type: " << frame.type() << endl;
                StabilizeVideo(frame_grey);
            }

            if(tracking){
                
            }




            key = waitKey(wait_ms);  // Capture Keyboard stroke
        }
    }

}









