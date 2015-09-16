// WindingControl.cpp
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//

#include "WindingControl.h"

#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>


#include <string>
#include <iostream>

using namespace std;
using namespace cv;
using namespace mvIMPACT::acquire;

Device* OpenCam(){

	DeviceManager devMgr;  // mvImpact implementation

    Device* pDev(nullptr);

    pDev = getDeviceFromUserInput( devMgr );

    if ( !pDev ) {
        cout << "Default device not found. Maybe pick one frome the list?" << endl;
        pDev = getDeviceFromUserInput( devMgr );
        if ( !pDev ) {
            cout << "Unable to continue!";
            cout << "Press [ENTER] to end the application" << endl;
            cin.get();
            exit(EXIT_FAILURE);
        }
    }

    cout << "Initialising the device. This might take some time..." << endl;
    try {
        pDev->open();
    }
    catch( const ImpactAcquireException& e ) {
        // this e.g. might happen if the same device is already opened in another process...
        cout << "An error occurred while opening the device " << pDev->serial.read() << " (error code: " << e.getErrorCode() << "). Press any key to end the application..." << endl;
        cout << "Press [ENTER] to end the application" << endl;
        cin.get();
        exit(EXIT_FAILURE);
    }

    cout << "The device " << pDev->serial.read() << " has been opened." << endl;

    return pDev;
}

void ShowPictureOfCamera(){

	int CAM_HEIGHT, CAM_WIDTH;

    initModule_nonfree();  // for licensed FeatureDetectors

    // different classes for getting the video input from file/cam/mvImpact
    VideoCapture cam;  // OpenCV wrapper for upnp cameras or video files

    DeviceManager devMgr;  // mvImpact implementation, device manager
    Device* pDev(nullptr);  // actual device
    // if the device manager is deleted all device pointers will be invalidated
    Ptr<FunctionInterface> fi;  // interface to get the frames from
    const int iMaxWaitTime_ms = 800000;  // timeout to wait for a frame
    int requestNr(0);  // handle frames by number and error codes
    int image_type(CV_8UC1);  // Mat type when using mvImpact camera

    Request* pRequest(nullptr), *pPreviousRequest(nullptr);  // pointers to the actual frames/requests

    Mat whole(Size(0, 0), CV_8UC3);  // container for the 'whole' video frame, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels

    cout << "Probing for devices..." << endl;
    pDev = getDeviceFromUserInput( devMgr );  // prints cnnected devices and lets the user choose

    if ( !pDev ) {
        cout << "Device not found. Try again" << endl;
        pDev = getDeviceFromUserInput( devMgr );  // second try
        if ( !pDev ) {  // if that failed exit
            cout << "Unable to continue!";
            cout << "Press [ENTER] to end the application" << endl;
            cin.get();
            exit(EXIT_FAILURE);
        }
    }

    cout << "Initialising the device. This might take some time..." << endl;
    try {
        pDev->open();  // try to open the device and exit if this failes for some reason
    }
    catch( const ImpactAcquireException& e ) {
        // this e.g. might happen if the same device is already opened by another process...
        cout << "An error occurred while opening the device " << pDev->serial.read() << " (error code: " << e.getErrorCode() << "). Press any key to end the application..." << endl;
        cout << "Press [ENTER] to end the application" << endl;
        cin.get();
        exit(EXIT_FAILURE);
    }

    cout << "The device " << pDev->serial.read() << " has been opened." << endl;

    // set the pixel depth to 8 bit
    modifyPropertyValue(ImageDestination(pDev).pixelFormat, "Mono8");

    // start an interface to the device
    fi = new FunctionInterface(pDev);
    // send a request to the default request queue of the device and wait for the result.
    fi->imageRequestSingle();
    // Start the acquisition manually if this was requested(this is to prepare the driver for data capture and tell the device to start streaming data)
    if ( pDev->acquisitionStartStopBehaviour.read() == assbUser ) {
        TDMR_ERROR result = DMR_NO_ERROR;
        if ( (result = static_cast<TDMR_ERROR>(fi->acquisitionStart())) != DMR_NO_ERROR ) {
            cout << "'FunctionInterface.acquisitionStart' returned with an unexpected result: " << result
                 << "(" << ImpactAcquireException::getErrorCodeAsString(result) << ")" << endl;
        }
    }

    // wait for results from the default capture queue
    requestNr = fi->imageRequestWaitFor(iMaxWaitTime_ms);

    cout << requestNr << endl;


    // get first frame so that width and height are defined
    pRequest = fi->getRequest(requestNr);
    if ( !pRequest->isOK() ) {
        cout << "Error: " << pRequest->requestResult.readS() << endl;
        exit(EXIT_FAILURE);
    }

    // get frame properties
    CAM_WIDTH  = pRequest->imageWidth.read();
    CAM_HEIGHT = pRequest->imageHeight.read();
    int channelcnt = pRequest->imageChannelCount.read();
    int bytesPP    = pRequest->imageBytesPerPixel.read();

    cout << "image width: " << CAM_WIDTH << " , image height: " << CAM_HEIGHT << " , No channel: " << channelcnt << " , bytes per pixel: " << bytesPP << endl;
    // this should no longer occur due to the fact that the pixel depth should've been set to Mono8 = 1 byte ; if this is the case exit with failure
    if (bytesPP > 1) {
        cout << "Bytes per pixel should be 1! Please try to configure camera using wxPropView. EXIT" << endl;
        exit(EXIT_FAILURE);
    }

    // import the frame from the request into the OpenCV container
    whole = Mat(Size(CAM_WIDTH, CAM_HEIGHT), image_type, pRequest->imageData.read());

    // key pressed and time to wait at the end of the loop
    char key('a');
    int wait_ms(10);

    // open the main window, add a trackbar for wait_ms
    namedWindow("Video Frame", CV_WINDOW_NORMAL);
    // resizeWindow("Video Frame", CAM_WIDTH-305, CAM_HEIGHT);
    createTrackbar("waitKey duration:", "Video Frame", &wait_ms, 1000, nullptr);

    // output format
    cout << "Video format is h: " << CAM_HEIGHT << ", w: " << CAM_WIDTH << "\nLoop starts..." << endl;

    // Create loop for continous streaming
    while (key != 27 && key != 'q'){  // end loop when 'ESC' or 'q' is pressed

        // show the processed frame and update the trackbars position
        imshow("Video Frame", whole);  // show the results in windows
        setTrackbarPos("waitKey duration:", "Video Frame", wait_ms);

        key = waitKey(wait_ms);  // Capture Keyboard stroke

        // again as before capture the next frame either from file or from device
        if ( pPreviousRequest ) pPreviousRequest->unlock();
        // this image has been displayed thus the buffer is no longer needed...
        pPreviousRequest = pRequest;

        fi->imageRequestSingle();
        requestNr = fi->imageRequestWaitFor(iMaxWaitTime_ms);

        // check if the image has been captured without any problems
        if ( !fi->isRequestNrValid(requestNr) ) {
            // If the error code is -2119(DEV_WAIT_FOR_REQUEST_FAILED), the documentation will provide
            // additional information under TDMR_ERROR in the interface reference
            cout << "imageRequestWaitFor failed (" << requestNr << ", " << ImpactAcquireException::getErrorCodeAsString(requestNr) << ")"
                 << ", timeout value too small?" << endl;
            exit(EXIT_FAILURE);
        }

        pRequest = fi->getRequest(requestNr);
        if ( !pRequest->isOK() ) {
            cout << "Error: " << pRequest->requestResult.readS() << endl;
            exit(EXIT_FAILURE);
        }

        // import frame into OpenCV container
        whole = Mat(Size(CAM_WIDTH, CAM_HEIGHT), image_type, pRequest->imageData.read());

    }

    cout << "Loop ended. Releasing capture and destroying windows..." << endl;

    // close all windows
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.

    // release the capture/requests and clear the request queue
    cam.release();

}

void WriteVideoToFile(string outputName){ // funktioniert noch nicht

	int CAM_HEIGHT, CAM_WIDTH;

    initModule_nonfree();  // for licensed FeatureDetectors

    // different classes for getting the video input from file/cam/mvImpact
    VideoCapture cam;  // OpenCV wrapper for upnp cameras or video files

    DeviceManager devMgr;  // mvImpact implementation, device manager
    Device* pDev(nullptr);  // actual device
    // if the device manager is deleted all device pointers will be invalidated
    Ptr<FunctionInterface> fi;  // interface to get the frames from
    const int iMaxWaitTime_ms = 800000;  // timeout to wait for a frame
    int requestNr(0);  // handle frames by number and error codes
    int image_type(CV_8UC1);  // Mat type when using mvImpact camera

    Request* pRequest(nullptr), *pPreviousRequest(nullptr);  // pointers to the actual frames/requests

    Mat whole(Size(0, 0), CV_8UC3);  // container for the 'whole' video frame, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels

    cout << "Probing for devices..." << endl;
    pDev = getDeviceFromUserInput( devMgr );  // prints cnnected devices and lets the user choose

    if ( !pDev ) {
        cout << "Device not found. Try again" << endl;
        pDev = getDeviceFromUserInput( devMgr );  // second try
        if ( !pDev ) {  // if that failed exit
            cout << "Unable to continue!";
            cout << "Press [ENTER] to end the application" << endl;
            cin.get();
            exit(EXIT_FAILURE);
        }
    }

    cout << "Initialising the device. This might take some time..." << endl;
    try {
        pDev->open();  // try to open the device and exit if this failes for some reason
    }
    catch( const ImpactAcquireException& e ) {
        // this e.g. might happen if the same device is already opened by another process...
        cout << "An error occurred while opening the device " << pDev->serial.read() << " (error code: " << e.getErrorCode() << "). Press any key to end the application..." << endl;
        cout << "Press [ENTER] to end the application" << endl;
        cin.get();
        exit(EXIT_FAILURE);
    }

    cout << "The device " << pDev->serial.read() << " has been opened." << endl;

    // set the pixel depth to 8 bit
    modifyPropertyValue(ImageDestination(pDev).pixelFormat, "Mono8");

    // start an interface to the device
    fi = new FunctionInterface(pDev);
    // send a request to the default request queue of the device and wait for the result.
    fi->imageRequestSingle();
    // Start the acquisition manually if this was requested(this is to prepare the driver for data capture and tell the device to start streaming data)
    if ( pDev->acquisitionStartStopBehaviour.read() == assbUser ) {
        TDMR_ERROR result = DMR_NO_ERROR;
        if ( (result = static_cast<TDMR_ERROR>(fi->acquisitionStart())) != DMR_NO_ERROR ) {
            cout << "'FunctionInterface.acquisitionStart' returned with an unexpected result: " << result
                 << "(" << ImpactAcquireException::getErrorCodeAsString(result) << ")" << endl;
        }
    }

    // wait for results from the default capture queue
    requestNr = fi->imageRequestWaitFor(iMaxWaitTime_ms);

    // get first frame so that width and height are defined
    pRequest = fi->getRequest(requestNr);
    if ( !pRequest->isOK() ) {
        cout << "Error: " << pRequest->requestResult.readS() << endl;
        exit(EXIT_FAILURE);
    }

    // get frame properties
    CAM_WIDTH  = pRequest->imageWidth.read();
    CAM_HEIGHT = pRequest->imageHeight.read();
    int channelcnt = pRequest->imageChannelCount.read();
    int bytesPP    = pRequest->imageBytesPerPixel.read();

    cout << "image width: " << CAM_WIDTH << " , image height: " << CAM_HEIGHT << " , No channel: " << channelcnt << " , bytes per pixel: " << bytesPP << endl;
    // this should no longer occur due to the fact that the pixel depth should've been set to Mono8 = 1 byte ; if this is the case exit with failure
    if (bytesPP > 1) {
        cout << "Bytes per pixel should be 1! Please try to configure camera using wxPropView. EXIT" << endl;
        exit(EXIT_FAILURE);
    }

    // import the frame from the request into the OpenCV container
    whole = Mat(Size(CAM_WIDTH, CAM_HEIGHT), image_type, pRequest->imageData.read());

    // key pressed and time to wait at the end of the loop
    char key('a');
    int wait_ms(10);

    // open the main window, add a trackbar for wait_ms
    namedWindow("Video Frame", CV_WINDOW_NORMAL);
    // resizeWindow("Video Frame", CAM_WIDTH-305, CAM_HEIGHT);
    createTrackbar("waitKey duration:", "Video Frame", &wait_ms, 1000, nullptr);

    // output format
    cout << "Video format is h: " << CAM_HEIGHT << ", w: " << CAM_WIDTH << endl;

    // Container for saving a video
    VideoWriter writer("Out.avi", //Name of output file
               CV_FOURCC('M','J','P','G'), //-character code of codec used to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc.
               10, //framerate
               Size(CAM_WIDTH, CAM_HEIGHT));
   // writer.open(outputName, CV_FOURCC('F','M','P','4'), 20, Size(CAM_WIDTH, CAM_HEIGHT));

    // check if the output file ist opened; exit if this is not the case
    if (writer.isOpened()) {
        cout << "VideoWriter is opened. Writing videostream to file '" << outputName << "'" << endl;
    } else {
        cout << " VideoWriter couldn't be openend for file '" << outputName << "'. EXIT" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Loop starts..." << endl;

    // Create loop for continous streaming
    while (key != 27 && key != 'q'){  // end loop when 'ESC' or 'q' is pressed

    	cam >> whole;

    	//write image to writer
        writer.write(whole);

        // show the processed frame and update the trackbars position
        imshow("Video Frame", whole);  // show the results in windows
        setTrackbarPos("waitKey duration:", "Video Frame", wait_ms);

        key = waitKey(wait_ms);  // Capture Keyboard stroke

        // again as before capture the next frame either from file or from device
        if ( pPreviousRequest ) pPreviousRequest->unlock();
        // this image has been displayed thus the buffer is no longer needed...
        pPreviousRequest = pRequest;

        fi->imageRequestSingle();
        requestNr = fi->imageRequestWaitFor(iMaxWaitTime_ms);

        // check if the image has been captured without any problems
        if ( !fi->isRequestNrValid(requestNr) ) {
            // If the error code is -2119(DEV_WAIT_FOR_REQUEST_FAILED), the documentation will provide
            // additional information under TDMR_ERROR in the interface reference
            cout << "imageRequestWaitFor failed (" << requestNr << ", " << ImpactAcquireException::getErrorCodeAsString(requestNr) << ")"
                 << ", timeout value too small?" << endl;
            exit(EXIT_FAILURE);
        }

        pRequest = fi->getRequest(requestNr);
        if ( !pRequest->isOK() ) {
            cout << "Error: " << pRequest->requestResult.readS() << endl;
            exit(EXIT_FAILURE);
        }

        // import frame into OpenCV container
        whole = Mat(Size(CAM_WIDTH, CAM_HEIGHT), image_type, pRequest->imageData.read());

    }

    cout << "Loop ended. Releasing capture and destroying windows..." << endl;

    // close all windows
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.

    // release the capture/requests and clear the request queue
    cam.release();

}


 
void CaptureVideoFromFile(string VideoName){

	VideoCapture cap(VideoName); // open the video file for reading


    if( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         exit(EXIT_FAILURE);
    }

    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

    cout << "Frame per seconds : " << fps << endl;

    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    while(1)
    {
        Mat frame;

        bool bSuccess = cap.read(frame); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
                        cout << "Cannot read the frame from video file" << endl;
                       break;
        }

        imshow("MyVideo", frame); //show the frame in "MyVideo" window

        if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
       	{
                cout << "esc key is pressed by user" << endl; 
                break; 
       	}

    }
    

}