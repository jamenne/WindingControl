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

void ProzessFrame2(){

    //our sensitivity value to be used in the absdiff() function
    const static int SENSITIVITY_VALUE = 20;
    //size of blur used to smooth the intensity image output from absdiff() function
    const static int BLUR_SIZE = 10;

    //some boolean variables for added functionality
    bool objectDetected = false;
    //these two can be toggled by pressing 'd' or 't'
    bool debugMode = false;
    bool trackingEnabled = false;
    //pause and resume code
    bool pause = false;
    //set up the matrices that we will need
    //the two frames we will be comparing
    Mat frame1,frame2;
    //their grayscale images (needed for absdiff() function)
    Mat grayImage1,grayImage2;
    //resulting difference image
    Mat differenceImage;
    //thresholded difference image (for use in findContours() function)
    Mat thresholdImage;
    //video capture object.
    VideoCapture capture;

    while(1){

        //we can loop the video by re-opening the capture every time the video reaches its last frame

        capture.open("/home/e5a-labor/src/Winding Control/windingcontrol/out.mp4");

        if(!capture.isOpened()){
            cout<<"ERROR ACQUIRING VIDEO FEED\n";
            getchar();
            exit(EXIT_FAILURE);
        }

        //check if the video has reach its last frame.
        //we add '-1' because we are reading two frames from the video at a time.
        //if this is not included, we get a memory error!
        while(capture.get(CV_CAP_PROP_POS_FRAMES)<capture.get(CV_CAP_PROP_FRAME_COUNT)-1){

            //read first frame
            capture.read(frame1);
            //convert frame1 to gray scale for frame differencing
            cvtColor(frame1,grayImage1,COLOR_BGR2GRAY);
            //copy second frame
            capture.read(frame2);
            //convert frame2 to gray scale for frame differencing
            cvtColor(frame2,grayImage2,COLOR_BGR2GRAY);
            //perform frame differencing with the sequential images. This will output an "intensity image"
            //do not confuse this with a threshold image, we will need to perform thresholding afterwards.
            absdiff(grayImage1,grayImage2,differenceImage);
            //threshold intensity image at a given sensitivity value
            threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            if(debugMode==true){
                //show the difference image and threshold image
                imshow("Difference Image",differenceImage);
                imshow("Threshold Image", thresholdImage);
            }else{
                //if not in debug mode, destroy the windows so we don't see them anymore
                destroyWindow("Difference Image");
                destroyWindow("Threshold Image");
            }
            //blur the image to get rid of the noise. This will output an intensity image
            blur(thresholdImage,thresholdImage,Size(BLUR_SIZE,BLUR_SIZE));
            //threshold again to obtain binary image from blur output
            threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            if(debugMode==true){
                //show the threshold image after it's been "blurred"

                imshow("Final Threshold Image",thresholdImage);

            }
            else {
                //if not in debug mode, destroy the windows so we don't see them anymore
                destroyWindow("Final Threshold Image");
            }

            //if tracking enabled, search for contours in our thresholded image
            if(trackingEnabled){

                searchForMovement(thresholdImage,frame1);
            }

            //show our captured frame
            imshow("Frame1",frame1);
            //check to see if a button has been pressed.
            //this 10ms delay is necessary for proper operation of this program
            //if removed, frames will not have enough time to referesh and a blank 
            //image will appear.
            switch(waitKey(10)){

                case 27: //'esc' key has been pressed, exit program.
                    exit(EXIT_SUCCESS);
                case 116: //'t' has been pressed. this will toggle tracking
                    trackingEnabled = !trackingEnabled;
                    if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
                    else cout<<"Tracking enabled."<<endl;
                    break;
                case 100: //'d' has been pressed. this will debug mode
                    debugMode = !debugMode;
                    if(debugMode == false) cout<<"Debug mode disabled."<<endl;
                    else cout<<"Debug mode enabled."<<endl;
                    break;
                case 112: //'p' has been pressed. this will pause/resume the code.
                    pause = !pause;
                    if(pause == true){ cout<<"Code paused, press 'p' again to resume"<<endl;
                        while (pause == true){
                            //stay in this loop until 
                            switch (waitKey()){
                                //a switch statement inside a switch statement? Mind blown.
                            case 112: 
                                //change pause back to false
                                pause = false;
                                cout<<"Code Resumed"<<endl;
                                break;
                            }
                        }
                    }
            }
        }

        //release the capture before re-opening and looping again.
        capture.release();
    }

}

void ProzessFrame(){

    //our sensitivity value to be used in the absdiff() function
    const static int SENSITIVITY_VALUE = 20;
    //size of blur used to smooth the intensity image output from absdiff() function
    const static int BLUR_SIZE = 10;

    //some boolean variables for added functionality
    bool objectDetected = false;
    //these two can be toggled by pressing 'd' or 't'
    bool debugMode = false;
    bool trackingEnabled = false;
    //pause and resume code
    bool pause = false;
    //set up the matrices that we will need
    //the two frames we will be comparing
    Mat frame1,frame2;
    //their grayscale images (needed for absdiff() function)
    Mat grayImage1,grayImage2;
    //resulting difference image
    Mat differenceImage;
    //thresholded difference image (for use in findContours() function)
    Mat thresholdImage;
    //video capture object.
    VideoCapture capture;

    while(1){

        //we can loop the video by re-opening the capture every time the video reaches its last frame

        capture.open("/home/e5a-labor/src/Winding Control/windingcontrol/out.mp4");

        if(!capture.isOpened()){
            cout<<"ERROR ACQUIRING VIDEO FEED\n";
            getchar();
            exit(EXIT_FAILURE);
        }

        //check if the video has reach its last frame.
        //we add '-1' because we are reading two frames from the video at a time.
        //if this is not included, we get a memory error!
        while(capture.get(CV_CAP_PROP_POS_FRAMES)<capture.get(CV_CAP_PROP_FRAME_COUNT)-1){

            //read first frame
            capture.read(frame1);
            //convert frame1 to gray scale for frame differencing
            cvtColor(frame1,grayImage1,COLOR_BGR2GRAY);
            //copy second frame
            capture.read(frame2);
            //convert frame2 to gray scale for frame differencing
            cvtColor(frame2,grayImage2,COLOR_BGR2GRAY);
            //perform frame differencing with the sequential images. This will output an "intensity image"
            //do not confuse this with a threshold image, we will need to perform thresholding afterwards.
            absdiff(grayImage1,grayImage2,differenceImage);
            //threshold intensity image at a given sensitivity value
            threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            if(debugMode==true){
                //show the difference image and threshold image
                imshow("Difference Image",differenceImage);
                imshow("Threshold Image", thresholdImage);
            }else{
                //if not in debug mode, destroy the windows so we don't see them anymore
                destroyWindow("Difference Image");
                destroyWindow("Threshold Image");
            }
            //blur the image to get rid of the noise. This will output an intensity image
            blur(thresholdImage,thresholdImage,Size(BLUR_SIZE,BLUR_SIZE));
            //threshold again to obtain binary image from blur output
            threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
            if(debugMode==true){
                //show the threshold image after it's been "blurred"

                imshow("Final Threshold Image",thresholdImage);

            }
            else {
                //if not in debug mode, destroy the windows so we don't see them anymore
                destroyWindow("Final Threshold Image");
            }

            //if tracking enabled, search for contours in our thresholded image
            if(trackingEnabled){

                searchForMovement(thresholdImage,frame1);
            }

            //show our captured frame
            imshow("Frame1",frame1);
            //check to see if a button has been pressed.
            //this 10ms delay is necessary for proper operation of this program
            //if removed, frames will not have enough time to referesh and a blank 
            //image will appear.
            switch(waitKey(10)){

                case 27: //'esc' key has been pressed, exit program.
                    exit(EXIT_SUCCESS);
                case 116: //'t' has been pressed. this will toggle tracking
                    trackingEnabled = !trackingEnabled;
                    if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
                    else cout<<"Tracking enabled."<<endl;
                    break;
                case 100: //'d' has been pressed. this will debug mode
                    debugMode = !debugMode;
                    if(debugMode == false) cout<<"Debug mode disabled."<<endl;
                    else cout<<"Debug mode enabled."<<endl;
                    break;
                case 112: //'p' has been pressed. this will pause/resume the code.
                    pause = !pause;
                    if(pause == true){ cout<<"Code paused, press 'p' again to resume"<<endl;
                        while (pause == true){
                            //stay in this loop until 
                            switch (waitKey()){
                                //a switch statement inside a switch statement? Mind blown.
                            case 112: 
                                //change pause back to false
                                pause = false;
                                cout<<"Code Resumed"<<endl;
                                break;
                            }
                        }
                    }
            }
        }

        //release the capture before re-opening and looping again.
        capture.release();
    }

}

void searchForMovement(Mat thresholdImage, Mat &cameraFeed){
    //notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
    //to take the values passed into the function and manipulate them, rather than just working with a copy.
    //eg. we draw to the cameraFeed to be displayed in the main() function.

    //bounding rectangle of the object, we will use the center of this as its position.
    Rect objectBoundingRectangle = Rect(0,0,0,0);

    //we'll have just one object to search for
    //and keep track of its position.
    int theObject[2] = {0,0};

    bool objectDetected = false;
    Mat temp;
    thresholdImage.copyTo(temp);
    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    //find contours of filtered image using openCV findContours function
    //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
    findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)objectDetected=true;
    else objectDetected = false;

    if(objectDetected){
        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));
        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));
        int xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        int ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;

        //update the objects positions by changing the 'theObject' array values
        theObject[0] = xpos , theObject[1] = ypos;
    }
    //make some temp x and y variables so we dont have to type out so much
    int x = theObject[0];
    int y = theObject[1];
    
    //draw some crosshairs around the object
    circle(cameraFeed,Point(x,y),20,Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    line(cameraFeed,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

    //write the position of the object to the screen
    putText(cameraFeed,"Tracking object at (" + intToString(x)+","+intToString(y)+")",Point(x,y),1,1,Scalar(255,0,0),2);

    

}

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

void WriteVideoToFile(string outputName, Device* pDev){ // funktioniert noch nicht

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
    VideoWriter outputVideo("out.avi",CV_FOURCC('M','J','P','G'),10, Size(CAM_WIDTH,CAM_HEIGHT),false);

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

//int to string helper function
string intToString(int number){

    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}

//-----------------------------------------------------------------------------
void inplaceHorizontalMirror( void* pData, int height, size_t pitch )
//-----------------------------------------------------------------------------
{
    int upperHalfOfLines = height / 2; // the line in the middle (if existent) doesn't need to be processed!
    char* pLowerLine = static_cast<char*>( pData ) + ( ( height - 1 ) * pitch );
    char* pUpperLine = static_cast<char*>( pData );
    char* pTmpLine = new char[pitch];
    for( int y = 0; y < upperHalfOfLines; y++ )
    {
        memcpy( pTmpLine, pUpperLine, pitch );
        memcpy( pUpperLine, pLowerLine, pitch );
        memcpy( pLowerLine, pTmpLine, pitch );
        pUpperLine += pitch;
        pLowerLine -= pitch;
    }
    delete [] pTmpLine;
}