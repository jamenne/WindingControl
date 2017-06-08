// WindingControl.cpp
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//

#include "WindingControl.h"

//OpenCV stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <common/minmax.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire_GenICam.h>


#include <string>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;
using namespace mvIMPACT::acquire;
using namespace mvIMPACT::acquire::GenICam;


// Initializes the camera - working!
Device* OpenCam(DeviceManager &devMgr){

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

// Initializes the Image - working!
Mat InitializeImage(Device* pDev){

    Ptr<FunctionInterface> fi;  // interface to get the frames from
    const int iMaxWaitTime_ms = 8000;  // timeout to wait for a frame
    int requestNr(0);  // handle frames by number and error codes
    int image_type(CV_8UC1);  // Mat type when using mvImpact camera

    Request* pRequest(nullptr);  // pointers to the actual frames/requests

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
    int cam_width  = pRequest->imageWidth.read();
    int cam_height = pRequest->imageHeight.read();
    int channelcnt = pRequest->imageChannelCount.read();
    int bytesPP    = pRequest->imageBytesPerPixel.read();

    cout << "image width:\t" << cam_width << "\nimage height:\t" << cam_height << "\nNo channel:\t" << channelcnt << "\nbytes per pixel:\t" << bytesPP << endl;
    // this should no longer occur due to the fact that the pixel depth should've been set to Mono8 = 1 byte ; if this is the case exit with failure
    if (bytesPP > 1) {
        cout << "Bytes per pixel should be 1! Please try to configure camera using wxPropView. EXIT" << endl;
        exit(EXIT_FAILURE);
    }


    // import the frame from the request into the OpenCV container
    Mat frame = Mat(Size(cam_width, cam_height), image_type, pRequest->imageData.read());

    return frame;
    /*
    while(1) {
        namedWindow("Video Frame", CV_WINDOW_AUTOSIZE);
        imshow("Video Frame", frame);

        if(waitKey(0) == 27) break;
    }

    // close all windows
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.
    */
}

// Asks for a single image - working!
Mat ImageRequestSingle(Device* pDev){ // Don't use for an continuous capture -> too slow

    FunctionInterface fi(pDev);

    // send a request to the default request queue of the device and wait for the result.
    fi.imageRequestSingle();
    manuallyStartAcquisitionIfNeeded( pDev, fi );

    // Define the Image Result Timeout (The maximum time allowed for the Application
    // to wait for a Result). Infinity value:-1
    const int iMaxWaitTime_ms = -1;   // USB 1.1 on an embedded system needs a large timeout for the first image.

    // wait for results from the default capture queue.
    int requestNr = fi.imageRequestWaitFor( iMaxWaitTime_ms );
    manuallyStopAcquisitionIfNeeded( pDev, fi );

    // check if the image has been captured without any problems.
    if( !fi.isRequestNrValid( requestNr ) )
    {
        // If the error code is -2119(DEV_WAIT_FOR_REQUEST_FAILED), the documentation will provide
        // additional information under TDMR_ERROR in the interface reference
        cout << "imageRequestWaitFor failed (" << requestNr << ", " << ImpactAcquireException::getErrorCodeAsString( requestNr ) << ")"
             << ", timeout value too small?" << endl;
        exit(EXIT_FAILURE);
    }

    const Request* pRequest = fi.getRequest( requestNr );
    if( !pRequest->isOK() )
    {
        cout << "Error: " << pRequest->requestResult.readS() << endl;
        // if the application wouldn't terminate at this point this buffer HAS TO be unlocked before
        // it can be used again as currently it is under control of the user. However terminating the application
        // will free the resources anyway thus the call
        // fi.imageRequestUnlock( requestNr );
        // can be omitted here.
        exit(EXIT_FAILURE);
    }

    cout << "Image captured( " << pRequest->imagePixelFormat.readS() << " " << pRequest->imageWidth.read() << "x" << pRequest->imageHeight.read() << " )" << endl;
    
    // import frame into OpenCV container
    Mat frame(1200,1600,CV_8U);
    frame = Mat(frame.size(), frame.type(), pRequest->imageData.read());

    return frame;
}

// shows the image of the camera continously - working!
void ShowPictureOfCamera(Device* pDev){

    int CAM_HEIGHT, CAM_WIDTH;

    Ptr<FunctionInterface> fi;  // interface to get the frames from
    const int iMaxWaitTime_ms = 8000;  // timeout to wait for a frame
    int requestNr(0);  // handle frames by number and error codes
    int image_type(CV_8UC1);  // Mat type when using mvImpact camera

    Request* pRequest(nullptr), *pPreviousRequest(nullptr);  // pointers to the actual frames/requests

    Mat whole(Size(0, 0), CV_8UC3);  // container for the 'whole' video frame, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels

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
}

// saves images as a video to file - working!
void WriteVideoToFile(string outputName, Device* pDev){ 

    int CAM_HEIGHT, CAM_WIDTH;

    Ptr<FunctionInterface> fi;  // interface to get the frames from
    const int iMaxWaitTime_ms = 8000;  // timeout to wait for a frame
    int requestNr(0);  // handle frames by number and error codes
    int image_type(CV_8UC1);  // Mat type when using mvImpact camera

    Request* pRequest(nullptr), *pPreviousRequest(nullptr);  // pointers to the actual frames/requests

    Mat whole(Size(0, 0), CV_8UC3);  // container for the 'whole' video frame, CV_8UC3 means we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels

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

    // it's important to set the last parameter bool isColor=false, otherwise he'll expect a colored image which doesn't exist!
    VideoWriter outputVideo(outputName,CV_FOURCC('M','J','P','G'),10, Size(CAM_WIDTH,CAM_HEIGHT),false);

    if (outputVideo.isOpened()) {
        cout << "VideoWriter is opened. Writing videostream to file '" << outputName << "'" << endl;
    } 
    else{
        cout << " VideoWriter couldn't be openend for file '" << outputName << "'. EXIT" << endl;
        exit(EXIT_FAILURE);
    }

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

        // write Video to file
        outputVideo.write(whole);

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

}

// reads a captured video from a file - working! 
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

// sets an chosen Output to an On or Off state - working!
void SetOutput( Device* pDev, int Output, bool On ){
//-----------------------------------------------------------------------------
//       This device has 6 DigitalIOs
// ------------------------------------------
// IO 0:   Type: Output    Current state: OFF
// IO 1:   Type: Output    Current state: OFF
// IO 2:   Type: Output    Current state: OFF
// IO 3:   Type: Output    Current state: OFF
// IO 4:   Type: Input     Current state: OFF
// IO 5:   Type: Input     Current state: OFF
// ------------------------------------------
    try{

        DigitalIOControl dioc( pDev );

        //Count the number of Digital IOs
        const unsigned int IOCount = dioc.lineSelector.dictSize();

        const unsigned int index = static_cast<unsigned int>( Output );

        if( ( index >= IOCount ))
        {
            cout << "Invalid selection" << endl;
            exit(EXIT_FAILURE);
        }

        //Define the IO we are interested in
        dioc.lineSelector.write( index );

        //check whether selected IO is an Output or an Input
        if( dioc.lineMode.readS() == "Output" )
        {
            if (On)
            {
                dioc.lineInverter.write( bTrue );
            }

            else dioc.lineInverter.write( bFalse );
        }

        else{
            
            cout << "IO " << index << " is a '" << dioc.lineMode.readS() << "'!" << endl;
        }
    }

    catch( const ImpactAcquireException& e )
    {
        cout << endl;
        cout << " An mvIMPACT Acquire Exception occurred:" << e.getErrorCodeAsString() << endl;
        cout << endl;
    }

}