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
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp> // for video stabilisation
#include <opencv2/nonfree/nonfree.hpp>

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
Mat ImageRequestSingle(Device* pDev, Mat &frame){ // Don't use for an continuous capture -> too slow

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

    //cout << "Image captured( " << pRequest->imagePixelFormat.readS() << " " << pRequest->imageWidth.read() << "x" << pRequest->imageHeight.read() << " )" << endl;
    
    // import frame into OpenCV container
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

////////////////////////  OBSOLETE /////////////////////////////////////



void ProzessFrame(Device* pDev, bool stab){

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

        /*if (stab)
        {
            stabilizeFrame(whole, "PYR_LK_OPTICALFLOW");
        }*/

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

void ProzessCapturedFrame(string VideoName, bool stab){

    VideoCapture cap; // open the video file for reading

    namedWindow("Original Video",CV_WINDOW_AUTOSIZE); //create a window called "Original Video"
    namedWindow("Stabilized Video",CV_WINDOW_AUTOSIZE); //create a window called "Stabilized Video"

    Mat frame;
    Mat stabframe;

    char key('a');
    int wait_ms(10);
    
    while(key != 27 && key != 'q'){

        cap.open(VideoName);

        if ( !cap.isOpened() ){  // if not success, exit program

         cout << "Cannot open the video file" << endl;
         exit(EXIT_FAILURE);
        }

        //check if the video has reach its last frame.
        //we add '-1' because we are reading two frames from the video at a time.
        //if this is not included, we get a memory error!
        while(cap.get(CV_CAP_PROP_POS_FRAMES)<cap.get(CV_CAP_PROP_FRAME_COUNT) && key != 27 && key != 'q'){

            bool bSuccess = cap.read(frame); // read a new frame from video

            if (!bSuccess) //if not success, break loop
            {
                cout << "Cannot read the frame from video file" << endl;
                break;
            }

            if (stab)
            {
                stabilizeFrame(frame, stabframe, "PYR_LK_OPTICALFLOW");
                //imshow("Stabilized Video", stabframe); //show the frame in "Stabilized Video" window
            }

            imshow("Original Video", frame); //show the frame in "Original Video" window

            key = waitKey(wait_ms);  // Capture Keyboard stroke
        }
    }

}

void stabilizeFrame(Mat& whole, Mat& stabframe, string method) {
    static vector<Point2f> features_prev;
    vector<Point2f> features_curr;
    vector<uchar> status;
    vector<float> err;

    static Mat gray_prev;
    // static Mat stabframe_prev;
    Mat gray_curr;

    //Rect ROI(Point(CAM_WIDTH*1./2-ROI_LEFT, CAM_HEIGHT*1./2-ROI_UPPER), Point(CAM_WIDTH*1./2+ROI_RIGHT, CAM_HEIGHT*1./2+ROI_LOWER));
    stabframe = whole.clone();//whole(ROI).clone();

    if (stabframe.channels() == 3)
        cvtColor(stabframe, gray_curr, CV_BGR2GRAY);  // compute grayscale image
    else
        gray_curr = stabframe;

    /*if (RESET) {  // if RESET is specified clear everything
        gray_prev.release();
        features_prev.clear();
    }*/
    
    if (!gray_prev.empty() && gray_prev.size() == gray_curr.size() && !features_prev.empty()) {
        // Feature matching using (sparse) optical flow of points using Lucas-Kanade-Method
        // void calcOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, OutputArray status, OutputArray err,
        // Size winSize=Size(21,21), int maxLevel=3, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags=0, double minEigThreshold=1e-4 )
        calcOpticalFlowPyrLK(gray_prev, gray_curr, features_prev, features_curr, status, err, Size(21,21), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 5e-4);
        clearBadFeatures(features_prev, features_curr, status, err);  // clear features with status==false

        try {
            if (!features_curr.empty()) stabilize(method, features_prev, features_curr, whole);  // stabilize frame using specified method and point pairs
        } catch(cv::Exception) {}
    }

    // store for next loop, possible because of static variable
    gray_curr.copyTo(gray_prev);
    // stabframe.copyTo(stabframe_prev);
    drawArrows(stabframe, features_curr, features_prev);
    vector<KeyPoint> keypoints = findFeatures(gray_curr, "PyramidHARRIS", features_prev);  // find new "prev" features in current image with specified featureDetector and store
    
    //clustering_approach(features_prev);  // experimental

    // drawPoints(stabframe, features_prev);
    drawKeypoints(stabframe, keypoints, stabframe, Scalar(255, 0, 255));
    if (!features_curr.empty()) drawPoints(stabframe, features_curr, Scalar(0, 255, 255));
}

void stabilize(string method, const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {

    /*
    if(method == "VIDEOSTAB"){
        stabilize_Videostab(feat_prev, feat_curr, whole);      
        exit(EXIT_SUCCESS);  
    }*/

    if(method == "PYR_LK_OPTICALFLOW"){
        stabilize_OpticalFlow(feat_prev, feat_curr, whole);
        exit(EXIT_SUCCESS);
    } 
    /*
    if (method == "RIGID_TRANSFORM"){
        stabilize_RigidTransform(feat_prev, feat_curr, whole);
        exit(EXIT_SUCCESS);
    }

    if(method == "FIND_HOMOGRAPHY"){
        stabilize_Homography(feat_prev, feat_curr, whole);
        exit(EXIT_SUCCESS);
    }*/

    else cout << "Chosen stationizing method no. " << method << " not associated with an implementation." << endl;
   
}


void stabilize_OpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    Point shift = getOpticalFlow(feat_prev, feat_curr);

    Mat trans_mat = (Mat_<float>(2, 3) << 1, 0, -shift.x, 0, 1, -shift.y);  // Creates a translation matrix with the mean optical flow
    warpAffine(whole, whole, trans_mat, whole.size());  // translates the whole frame with the matrix
}
/*
void stabilize_RigidTransform(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    static Mat prevMotions = Mat::eye(3, 3, CV_64F);

    if (RESET) prevMotions = Mat::eye(3, 3, CV_64F);

    Mat M = estimateRigidTransform(feat_prev, feat_curr, false);

    M.resize(3, 0);
    M.at<double>(2, 2) = 1;
    M.at<double>(0, 0) = M.at<double>(1, 1) = 1;
    M.at<double>(1, 0) = M.at<double>(0, 1) = 0;

    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);
}

void stabilize_Homography(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    static Mat prevMotions = Mat::eye(3, 3, CV_64F);

    if (RESET) prevMotions = Mat::eye(3, 3, CV_64F);

    Mat M = findHomography(feat_prev, feat_curr, CV_RANSAC, 1);
    // Mat M = findHomography(feat_prev, feat_curr, CV_LMEDS);

    M.at<double>(2, 2) = 1;
    M.at<double>(2, 1) = M.at<double>(2, 0) = 0;
    M.at<double>(0, 0) = M.at<double>(1, 1) = 1;
    M.at<double>(1, 0) = M.at<double>(0, 1) = 0;

    double a = M.at<double>(0, 2);
    if (a*a > 100) M.at<double>(0, 2) = 0;
    a = M.at<double>(1, 2);
    if (a*a > 100) M.at<double>(1, 2) = 0;

    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);
}



void stabilize_Videostab(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    // static Ptr<OnePassStabilizer> onePassStabilizer = new OnePassStabilizer();

    static Mat prevMotions = Mat::eye(3, 3, CV_32F);

    //if (RESET) {
    //    prevMotions = Mat::eye(3, 3, CV_32F);
    //} else {
        Mat M = cv::videostab::estimateGlobalMotionRobust(feat_prev, feat_curr, cv::videostab::TRANSLATION, cv::videostab::RansacParams(RANSAC_SIZE, RANSAC_THRESH*1./100, RANSAC_EPS*1./100, RANSAC_PROB*1./100));
        // Mat M = cv::videostab::estimateGlobalMotionLeastSquares(feat_prev, feat_curr, cv::videostab::TRANSLATION, 0);

        float a = M.at<float>(0, 2);
        if (a*a > 100) M.at<float>(0, 2) = 0;
        a = M.at<float>(1, 2);
        if (a*a > 100) M.at<float>(1, 2) = 0;

        prevMotions = M * prevMotions;
    //}
    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);

    // k++;
}*/


void clearBadFeatures(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, vector<uchar>& stat, vector<float>& err) {
    // only applicable if stat and feat_curr is defined (through calcOpticalFlowPyrLK)
    assert(feat_prev.size() == feat_curr.size() && feat_prev.size() == stat.size());  // and the two features sets are the same size

    auto iter_1 = feat_prev.begin();
    auto iter_2 = feat_curr.begin();
    auto found = stat.begin();
    auto it_err = err.begin();

    while (iter_1 < feat_prev.end()) {
        if (!(static_cast<bool>(*found))) {
            // if status==false the feature was not found and feat_curr does not hold a proper point
            iter_1 = feat_prev.erase(iter_1);
            iter_2 = feat_curr.erase(iter_2);
            found = stat.erase(found);
            it_err = err.erase(it_err);
            continue;
        }
        if ( norm(*iter_1 - *iter_2) > 120 || *it_err > 1) {
            // remove features with an error greater than 1.0 because they are likely to be bad
            // and remove matched points with a distance greater that 120 for the same reason
            iter_1 = feat_prev.erase(iter_1);
            iter_2 = feat_curr.erase(iter_2);
            found = stat.erase(found);
            it_err = err.erase(it_err);
            continue;
        }
        ++iter_1;
        ++iter_2;
        ++found;
        ++it_err;
    }
}

vector<KeyPoint> findFeatures(Mat& img, string method, vector<Point2f>& feat){  // find good tracking Features in frame 'img' (grayscale)
    vector<KeyPoint> keypoints;

    Ptr<FeatureDetector> detector = FeatureDetector::create(method);
    /*Ptr<FeatureDetector> FeatureDetector::create(const string& detectorType)
    Parameters: 

    detectorType – Feature detector type.

    The following detector types are supported:
    
        "FAST" – FastFeatureDetector
        "STAR" – StarFeatureDetector
        "SIFT" – SIFT (nonfree module)
        "SURF" – SURF (nonfree module)
        "ORB" – ORB
        "BRISK" – BRISK
        "MSER" – MSER
        "GFTT" – GoodFeaturesToTrackDetector
        "HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
        "Dense" – DenseFeatureDetector
        "SimpleBlob" – SimpleBlobDetector
    
    Also a combined format is supported: feature detector adapter name ( "Grid" – GridAdaptedFeatureDetector, "Pyramid" – 
    PyramidAdaptedFeatureDetector ) + feature detectorname (see above), for example: "GridFAST", "PyramidSTAR" . */

    detector->detect(img, keypoints);

    KeyPoint::convert(keypoints, feat);  // store as Point2f
    return keypoints;
}

void drawArrows(Mat& drawframe, const vector<Point2f>& feat_1, const vector<Point2f>& feat_2) {
    assert(feat_1.size() == feat_2.size());  // zeichne Pfeile zwischen den zugeordneten Paaren von Punkten

    for (unsigned int i = 0; i < feat_1.size(); i++) {
        arrowedLine(drawframe, feat_1[i], feat_2[i], Scalar(255, 0, 0), 1, CV_AA, 0, 0.3);
    }
}

void drawPoints(Mat& drawframe, const vector<Point2f>& feat, Scalar s) {
    if (feat.empty()) cout << "Vector of Point2f \"feat\" empty." << endl;
    for (const Point2f& x : feat) {
        circle(drawframe, x, 2, s, -1, CV_AA, 0);
    }
}

Point2f getOpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr) {  // calculate mean differences of the feature points
    assert(feat_prev.size() == feat_curr.size());

    // Averaging approach
    float x_diff_mean(0), y_diff_mean(0);
    int n(0);


    for (unsigned int i = 0; i < feat_prev.size(); i++) {
        // Averaging approach
        n++;
        x_diff_mean += feat_curr[i].x - feat_prev[i].x;
        y_diff_mean += feat_curr[i].y - feat_prev[i].y;
    }

    // Averaging approach
    return Point2f(x_diff_mean/n, y_diff_mean/n);
}



