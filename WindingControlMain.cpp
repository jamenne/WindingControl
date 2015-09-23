// WindingControlMain.cpp
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//
#include "WindingControl.h"
#include <iostream>

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>

using namespace std;

int main()
{	
	
	DeviceManager devMgr;

	Device* pDev = OpenCam(devMgr);
	WriteVideoToFile("Test.avi", pDev);

	//ShowPictureOfCamera(pDev);

    /*Mat whole(Size(0, 0), CV_8UC3);  // container for the 'whole' video frame, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels
	whole=InitializeImage(pDev);

	char key('a');
    int wait_ms(10);

    namedWindow("Video Frame", CV_WINDOW_AUTOSIZE);
    imshow("Video Frame", whole);

   while(key != 27 && key != 'q') {

        key = waitKey(wait_ms);

        whole = RequestNewImage(pDev, whole);

        imshow("Video Frame", whole);

    }

    // close all windows
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.*/

    //ProzessFrame();

	return 0;
}