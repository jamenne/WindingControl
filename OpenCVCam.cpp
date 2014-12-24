/*  Author: Philipp Zander
    at TU Dortmund, Physik, Experimentelle Physik 5
    year: 2014
*/
#include "TH1D.h"

#include "math.h"
#include <iostream>  // standard I/O
#include <string>  // strings
#include <sstream>  // for 
#include <fstream>  // reading/writing calibration files
#include <vector>

#include "opencv2/highgui/highgui.hpp"  // GUI windows and trackbars
#include "opencv2/video/video.hpp"  // camera/video classes, imports background_segm

using namespace cv;
using namespace std;

// global variables ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
double cam_height, cam_width;
int bw_tresh, canny_tresh, center_tresh, min_dist, min_radius, max_radius, accumulator_res;
int roi_lower, roi_upper, roi_left, roi_right, number_of_features = 200;
char key;
string videoName, inputName, outputName;
stringstream s;

vector<Vec3f> circles;  // Vektor für die gefundenen Kreise
vector<Point2f> corners_1, corners_2;
// vector<Point> current, next;  // Vektoren für 
vector<vector<Point>> contours;  // Vektor für gefundene Konturen
vector<unsigned char> status;
vector<float> err;

Mat whole, frame, fgMOG, fgMOG2, gray, binary, median, gauss, edges, prevGray;

BackgroundSubtractorMOG MOG;  // MOG approach 
BackgroundSubtractorMOG2 MOG2;  // MOG2 approach


// function declarations –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void usage() {  // display the program usage
    cout << "This Program was written at the\033[1m  Chair for Experimental Physics V\033[0m at the\033[1m TU Dortmund\033[0m" << endl;
    // cout << "If not provided an input file it will compute the stream from the main camera attached to your system at default. ";
    cout << "Press ESC or q to exit the program" << endl << "Usage: " << endl;
    cout << "\t\t >> Run from /videos/**/ <<" << endl;
    cout << "\t -input    \t[-i] \t **Not implemented** Input video file" << endl;
    cout << "\t -calibrate\t[-c] \t Calibrate the main parameters of the program. " << endl;
    cout << "\t\t\t\t If used there should be a '/calib' dir present." << endl;
    cout << "\t -help     \t[-h] \t Display this message" << endl;
}

void processImage();
void showOpticalFlow();
void showCircles();
void showContours();
void doNothing(int, void*) {}  // dummy function to pass to trackbars
void correctValues(int, void*);  // setting correct values for some parameters
void setDefaults();  // set default values for all parameters
void createCalibWindow();  // create the window where the calibration trackbars should be displayed (calibrate==true)
void readCalib();  // read the calibration parameters from file
void writeCalib();  // write the calibration parameters to another file
void showAll(); // function to display all the frames

// main ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
int main(int argc, char *argv[]) {
    bool calibrate = false, input = true;
    videoName  = "./videos/frame_%03d.jpg";
    inputName  = "./calib/calibration00.txt";

    if (argc > 2) {
        cout << argc << " is not a valid number of arguments. EXIT" << endl;
        usage();
        exit(EXIT_FAILURE);
    } else {
        if (argc == 2) {
            if (string(argv[1]) == string("-calibrate") || string(argv[1]) == string("-c")) {
                calibrate = true;
            } else {
                if (string(argv[1]) == string("-help") || string(argv[1]) == string("-h")) {
                    usage();
                    exit(EXIT_SUCCESS);
                } else {
                    cout << argv[1] << " is not a valid argument. EXIT" << endl;
                    usage();
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    // VideoCapture cam(CV_CAP_ANY);  // Capture using any camera connected
    VideoCapture cam(videoName);  // Load video file or a sequence of images

    if (!cam.isOpened()) {
        cout << "capture is not open. BREAK" << endl;
        exit(EXIT_FAILURE);
    }

    cam >> whole; // Create first frame from capture/video

    // get the format of the video/image stream
    cam_height = cam.get(CV_CAP_PROP_FRAME_HEIGHT);
    cam_width  = cam.get(CV_CAP_PROP_FRAME_WIDTH);

    cout << "Video format is h: " << cam_height << ", w: " << cam_width << endl;

    if (input)
        readCalib();
    else
        setDefaults();

    if (calibrate) {
        createCalibWindow();
    }

    while (key != 27 && key != 'q') {  // Create loop for streaming
      // end loop when 'ESC' or 'q' is pressed

        if (whole.empty()) {
            cam.open(videoName);
            cam >> whole;
        }

        processImage();
        showAll();

        key = waitKey(200);  // Capture Keyboard stroke

        cam >> whole;  // Get next frame
    }

    cam.release();
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.

    while (calibrate && key != 'n' && key != 'N') {
        cout << "Do you want to store the new calibration? (y/n) ";
        cin >> key;
        if (key == 'y' || key == 'Y') {
            cout << "Type the path of the new calibration file: (type '-d' for default file './calib/calibration00.txt')" << endl;
            cin >> outputName;
            if (outputName == string("-d"))
                outputName = "./calib/calibration00.txt";
            writeCalib();
            key = 'n';
        }
    }

    exit(EXIT_SUCCESS);
}

// actual doing something with the frames ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void processImage() {
    // define region of interest with parameters set by trackbars
    Rect ROI(Point(cam_width/2-roi_left, cam_height/2+roi_upper), Point(cam_width/2+roi_right, cam_height/2-roi_lower));
    frame = whole.operator()(ROI);  // extract ROI 

    // Background substraction of the live image using MOG and MOG2
    MOG.operator()(frame, fgMOG, 0.15);
    MOG2.operator()(frame, fgMOG2, 0.15);

    cvtColor(frame, gray, CV_BGR2GRAY);  // compute grayscale image
    threshold(gray, binary, bw_tresh, 255, THRESH_BINARY);  // compute b/w img

    GaussianBlur(binary, gauss, Size(3, 3), 3);
    medianBlur(binary, median, 5);

    // detect edges in the blurred binary image using Canny algorithm
    Canny(median, edges, canny_tresh, 2*canny_tresh, 5, true);

    /* detect circles in the blurred binary image using the Hough transformation
     1: src, 2: output array of vectors
     3: VV_HOUGH_GRADIENT: Definiert die Methode zum Detektieren der Kreise.
     4: dp (Double): Auflösung des Akkumulators. Wird für die Center-Detektion genutzt. Wenn 1, Akkumulator hat die 
        gleiche Auflösung wie das Inputbild, bei dp=2 der Akkumulator wird zweimal kleinere Höhe und Breite haben
     5: min_dist = Kleinste Distanz zwischen 2 Kreisen, je kleiner, desto mehr Kreise
     6: tresh_1: Obere Threshold für den Canny edge detector, je größer, desto weniger Kreise werden gefunden
     7: tresh_2: Threshold für Center Detection
     8: min_radius = 0: kleinster Radius, falls unbekannt, setze gleich 0
     9: max_radius = 0: Maximum Radius, falls unbekannt, setze gleich 0
   */
    HoughCircles(median, circles, CV_HOUGH_GRADIENT, accumulator_res, min_dist, 2*canny_tresh, center_tresh, min_radius, max_radius);
    showCircles();
    
    // findContours(binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    // showContours();

    if (prevGray.size() == gray.size()) {
        goodFeaturesToTrack(prevGray, corners_1, number_of_features, .1, .01);
        calcOpticalFlowPyrLK(prevGray, gray, corners_1, corners_2, status, err);
        showOpticalFlow();
    }

    gray.copyTo(prevGray);
    rectangle(whole, ROI, Scalar(0, 255, 0), 1, 8, 0);  // draw processed region of interest on whole image
}

void showCircles() {
    for (const Vec3f& circ: circles) {
        // speichert die ersten beiden Einträge von vector<Vec3f> circles in center und den dritten Wert in radius
        int x = circ[0], y = circ[1], radius = circ[2];
        Point center(x, y);

        circle(frame, center, radius, Scalar(0, 0, 255), 1, 8, 0);  // draw circle
        circle(frame, center, 2, Scalar(0, 0, 255), -1, 8, 0);  // draw center
    }
}

void showContours() {
    int i = 0;
    for(const vector<Point>& cont: contours)
    {

        Mat pointsf;
        Mat(cont).convertTo(pointsf, CV_32F);

        try{
            RotatedRect box = fitEllipse(pointsf);

            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
                continue;
            drawContours(frame, contours, i, Scalar::all(255), 1, 8);

            // if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*30 )
            //     continue;

            ellipse(frame, box, Scalar(0, 0, 255), 1, CV_AA);
            ellipse(frame, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, CV_AA);
            // Point2f vtx[4];
            // box.points(vtx);
            // for( int j = 0; j < 4; j++ )
            //     line(frame, vtx[j], vtx[(j+1)%4], Scalar(0,255,0), 1, CV_AA);
        }catch(Exception& e){}
    }
}

bool drawArrow(Point2f& a, Point2f& b) {
    double angle = atan2( (double) a.y - b.y, (double) a.x - b.x );
    double hypotenuse = sqrt( pow(a.y - b.y, 2) + pow(a.x - b.x, 2) );
    if (hypotenuse > 20)
        return false;
    
    /* Here we lengthen the arrow by a factor of three */
    // b.x = (int) (a.x - 3 * hypotenuse * cos(angle));
    // b.y = (int) (a.y - 3 * hypotenuse * sin(angle));
    /* Now we draw the main line of the arrow */
    /* "frame" is the frame to draw on.
     * "p" is the point where the line begins.
     * "q" is the point where the line stops.
     * "CV_AA" means antialiased drawing.
     * "0" means no fractional bits in the center cooridinate or radius.
    */
    line( frame, a, b, Scalar(255,0,0), 1, CV_AA, 0 );
    /* Now draw the tips of the arrow. I do some scaling so that the
     * tips look proportional to the main line of the arrow.
    */
    a.x = (int) (b.x + hypotenuse * cos(angle + CV_PI / 4));
    a.y = (int) (b.y + hypotenuse * sin(angle + CV_PI / 4));
    line( frame, a, b, Scalar(255,0,0), 1, CV_AA, 0 );
    a.x = (int) (b.x + hypotenuse * cos(angle - CV_PI / 4));
    a.y = (int) (b.y + hypotenuse * sin(angle - CV_PI / 4));
    line( frame, a, b, Scalar(255,0,0), 1, CV_AA, 0 );

    return true;
}

void showOpticalFlow(){
    int n = 0;
    double x_diff_mean = 0, y_diff_mean = 0;
    for(int i = 0; i < number_of_features; i++) {
        /* If Pyramidal Lucas Kanade didn't really find the feature, skip it */
        if ( status[i] == '0' ) 
            continue;

        /* Let's make the flow field look nice with arrows */
        /* The arrows will be a bit too short for a nice visualization because of the high framerate
         * (ie: there's not much motion between the frames). So let's lengthen them by a factor of 3.
        */
        // Point p,q;
        // p.x = (int) corners_1[i].x;
        // p.y = (int) corners_1[i].y;
        // q.x = (int) corners_2[i].x;
        // q.y = (int) corners_2[i].y;

        if (drawArrow(corners_1[i],corners_2[i])) {
            n++;
            x_diff_mean += corners_2[i].x - corners_1[i].x;
            y_diff_mean += corners_2[i].y - corners_1[i].y;
        }
    }

    s.str("");
    s << "x_diff_mean: " << x_diff_mean/n;
    putText(whole, s.str(), Point(15,30), FONT_HERSHEY_SIMPLEX, 1, Scalar::all(0), 2);
    s.str("");
    s << "y_diff_mean: " << y_diff_mean/n;
    putText(whole, s.str(), Point(15,60), FONT_HERSHEY_SIMPLEX, 1, Scalar::all(0), 2);
}

// calibration functions –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void correctValues(int, void*) {  // setting correct (>0) value for some parameters
    if (canny_tresh == 0)
        canny_tresh++;
    if (center_tresh == 0)
        center_tresh++;
    if (accumulator_res == 0)
        accumulator_res++;
    if (roi_lower < 10)
        roi_lower = 10;
    if (roi_upper < 10)
        roi_upper = 10;
    if (roi_left < 10)
        roi_left = 10;
    if (roi_right < 10)
        roi_right = 10;
    if (min_dist < 15)  // not necessarily, but a smaller value slow down the program
        min_dist = 15;

    setTrackbarPos("Center treshhold:", "Calibration window", center_tresh);
    setTrackbarPos("Canny lower treshhold:", "Calibration window", canny_tresh);
    setTrackbarPos("Reciprocal accumulator resolution:", "Calibration window", accumulator_res);
    setTrackbarPos("Lower boundary of ROI:", "Calibration window", roi_upper);
    setTrackbarPos("Upper boundary of ROI:", "Calibration window", roi_lower);
    setTrackbarPos("Left boundary of ROI: ", "Calibration window", roi_left);
    setTrackbarPos("Right boundary of ROI:", "Calibration window", roi_right);
    setTrackbarPos("Minimum distance between circles:", "Calibration window", min_dist);
}

void createCalibWindow() {  // create the window where the calibration trackbars should be displayed (calibrate==true)
    namedWindow("Calibration window", WINDOW_NORMAL);
    createTrackbar("Lower boundary of ROI:", "Calibration window", &roi_upper, cam_height/2, correctValues);
    createTrackbar("Upper boundary of ROI:", "Calibration window", &roi_lower, cam_height/2, correctValues);
    createTrackbar("Left boundary of ROI: ", "Calibration window", &roi_left,  cam_width/2,  correctValues);
    createTrackbar("Right boundary of ROI:", "Calibration window", &roi_right, cam_width/2,  correctValues);

    createTrackbar("B/W treshhold:", "Calibration window", &bw_tresh, 255, nullptr);
    createTrackbar("Canny lower treshhold:", "Calibration window", &canny_tresh, 1000, correctValues);
    createTrackbar("Center treshhold:", "Calibration window", &center_tresh, 100, correctValues);
    createTrackbar("Reciprocal accumulator resolution:", "Calibration window", &accumulator_res, 5, correctValues);
    createTrackbar("Minimum distance between circles:", "Calibration window", &min_dist, 255, correctValues);
    createTrackbar("Minimum radius of circles:", "Calibration window", &min_radius, 100, nullptr);
    createTrackbar("Maximum radius of circles:", "Calibration window", &max_radius, 100, nullptr);
}

void showAll() {  // function to display all the frames
    imshow("Video Frame", whole);
    // imshow("Region of interest", frame);

    // imshow("Foreground Mask MOG", fgMOG);
    // imshow("Foreground Mask MOG2", fgMOG2);

    imshow("Grayscale image", gray);
    imshow("Binary image", binary);

    imshow("Median blurred image", median);
    imshow("Gaussian blur image", gauss);

    imshow("Edges", edges);
}

// reading an writing from/to the calibration files ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void setDefaults() {  // set default values for all parameters
    roi_upper = cam_height*1./7.2;
    roi_lower = cam_height*1./6;
    roi_left  = cam_width/2 - 10;
    roi_right = cam_width*1./2.5;
    bw_tresh = 140;
    canny_tresh = 100;
    center_tresh = 8;
    accumulator_res = 1;
    min_dist = 50;
    min_radius = 15;
    max_radius = 25;
}

void readCalib() {
    string name;
    int value;

    bool all_set = false, set_min_radius = false, set_max_radius = false, set_accumulator_res = false;
    bool set_bw_tresh = false, set_canny_tresh = false, set_center_tresh = false, set_min_dist = false;
    bool set_roi_upper = false, set_roi_lower = false, set_roi_left = false, set_roi_right = false;

    ifstream inputFile(inputName);

    if (!inputFile.is_open()) {
        cout << "Input calibration file not readable. Setting default values for parameters." << endl;
        setDefaults();
        return;
    }

    while (!inputFile.eof()) {
        inputFile >> name >> value;
        if (name == string("bw_tresh") && !set_bw_tresh) {
            bw_tresh = value;
            set_bw_tresh = true;        
        }        
        if (name == string("canny_tresh") && !set_canny_tresh) {
            canny_tresh = value;
            set_canny_tresh = true;
        }
        if (name == string("center_tresh") && !set_center_tresh) {
            center_tresh = value;
            set_center_tresh = true;
        }
        if (name == string("min_dist") && !set_min_dist) {
            min_dist = value;
            set_min_dist = true;
        }
        if (name == string("min_radius") && !set_min_radius) {
            min_radius = value;
            set_min_radius = true;
        }
        if (name == string("max_radius") && !set_max_radius) {
            max_radius = value;
            set_max_radius = true;    
        }
        if (name == string("accumulator_res") && !set_accumulator_res) {
            accumulator_res = value;
            set_accumulator_res = true;    
        }
        if (name == string("roi_upper") && !set_roi_upper) {
            roi_upper = value;
            set_roi_upper = true;
        }
        if (name == string("roi_lower") && !set_roi_lower) {
            roi_lower = value;
            set_roi_lower = true;
        }
        if (name == string("roi_left") && !set_roi_left) {
            roi_left = value;
            set_roi_left = true;
        }
        if (name == string("roi_right") && !set_roi_right) {
            roi_right = value;
            set_roi_right = true;
        }
    }

    all_set = set_bw_tresh && set_canny_tresh && set_center_tresh && set_min_dist && set_min_radius && set_max_radius;
    all_set = all_set && set_accumulator_res && set_roi_upper && set_roi_lower && set_roi_left && set_roi_right;

    inputFile.close();

    if (all_set) {
        cout << "Succesfully read calibration parameters from file." << endl;
    } else {
        cout << "There was an error during the import of the calibration parameters. ";
        cout << "Parameters will be set to default values." << endl;
        setDefaults();
    }
}

void writeCalib() {
    ofstream outputFile(outputName);

    if (outputFile.is_open()) {
        outputFile << "roi_upper\t"    << roi_upper    << endl;
        outputFile << "roi_lower\t"    << roi_lower    << endl;
        outputFile << "roi_left\t"     << roi_left     << endl;
        outputFile << "roi_right\t"    << roi_right    << endl;
        outputFile << "bw_tresh\t"     << bw_tresh     << endl;
        outputFile << "canny_tresh\t"  << canny_tresh  << endl;
        outputFile << "center_tresh\t" << center_tresh << endl;
        outputFile << "min_dist\t"     << min_dist     << endl;
        outputFile << "min_radius\t"   << min_radius   << endl;
        outputFile << "max_radius\t"   << max_radius   << endl;
        outputFile << "accumulator_res\t" << accumulator_res << endl;
        outputFile.close();

        cout << "Succesfully written calibration parameters to file." << endl;
    } else {
        cout << "Unable to open output file." << endl;
    }
}













