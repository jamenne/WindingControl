/*  Copyright 2014 TU Dortmund, Physik, Experimentelle Physik 5
	
	Author: Philipp Zander

    build with openCV v2.4.10
        and ROOT v5.34/21
*/

#include <TH2D.h>
#include <TCanvas.h>
#include <opencv2/highgui/highgui.hpp>  // GUI windows and trackbars
#include <opencv2/video/video.hpp>  // camera/video classes, imports background_segm
#include <opencv2/calib3d/calib3d.hpp>  // for FindHomography
#include <opencv2/features2d/features2d.hpp>  // for FeatureDetection
#include <opencv2/videostab/stabilizer.hpp>
// #include <opencv2/nonfree/nonfree.hpp>

#include <boost/program_options.hpp>

#include <mvIMPACT_CPP/mvIMPACT_acquire.h>

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <sstream>  // used for putting text on the frames
#include <fstream>  // reading/writing configration files
#include <cmath>

using namespace cv;
using std::cout; using std::cin; using std::endl; using std::stringstream;
using std::vector; using std::ifstream; using std::ofstream;
namespace po = boost::program_options;

// global variables ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
int CAM_HEIGHT, CAM_WIDTH;
int BW_THRESH, CANNY_THRESH, CENTER_THRESH, MIN_DIST, MIN_RADIUS, MAX_RADIUS, ACCUMULATOR_RES;
int ROI_LOWER, ROI_UPPER, ROI_LEFT, ROI_RIGHT, VOTING_BINSIZE, RANSAC_SIZE;
int RANSAC_THRESH, RANSAC_EPS, RANSAC_PROB, MOG_LEARNINGRATE, NUM_OF_THREADS;
// int H_MIN, H_MAX, V_MIN, V_MAX, S_MIN, S_MAX;
// int B_MIN, B_MAX, G_MIN, G_MAX, R_MIN, R_MAX;

uint RESET;
bool ALARM;

float MEAN_FIBER_DISTANCE, MEAN_FIBER_DIAMETER, MEAN_X_LEFTOFPRIMARY, MEAN_X_RIGHTOFPRIMARY, MEAN_Y_ABOVEPRIMARY, MEAN_Y_BELOWPRIMARY;

enum StabilizationMethod {PYR_LK_OPTICALFLOW, RIGID_TRANSFORM, FIND_HOMOGRAPHY, VIDEOSTAB, PHASE_CORRELATION};

// function declarations – function used by main –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void stabilizeFrame(Mat&);
void processFrame(Mat&);
void correctValues(int, void*);  // setting correct values for some parameters
// void setDefaults();  // set default values for all parameters
void createconfigWindow();  // create the window where the configration trackbars should be displayed (configrate==true)
// void readconfig();  // read the configration parameters from file
void writeconfig(string);  // write the configration parameters to another file
void setConfig(po::options_description& config_desc) {
    cv::videostab::RansacParams translation = cv::videostab::RansacParams::translationMotionStd();
    config_desc.add_options()
        ("ROI.UPPER", po::value<int>(&ROI_UPPER) -> default_value(static_cast<int>(CAM_HEIGHT*1./7.2)), "Location of the 'Region of Interest' depending on")
        ("ROI.LOWER", po::value<int>(&ROI_LOWER) -> default_value(static_cast<int>(CAM_HEIGHT*1./6)),   "the input videostream dimensions (even the default")
        ("ROI.LEFT",  po::value<int>(&ROI_LEFT)  -> default_value(static_cast<int>(CAM_WIDTH*1./2.1)),  "values) to regulate the upper, lower and the")
        ("ROI.RIGHT", po::value<int>(&ROI_RIGHT) -> default_value(static_cast<int>(CAM_WIDTH*1./2.5)),  "left, right edge of the RoI\n")

        ("BW_THRESH",      po::value<int>(&BW_THRESH)      -> default_value(140), "Threshold for converting grayscale to binary image")
        ("NUM_OF_THREADS", po::value<int>(&NUM_OF_THREADS) -> default_value(8),   "Number of fiber threads visible in the camera - primary\n")
        // ("VOTING_BINSIZE",  po::value<int>(&VOTING_BINSIZE)-> default_value(5), "")

        ("RANSAC.SIZE",   po::value<int>(&RANSAC_SIZE)   -> default_value(translation.size),       "Parameters for the RANSAC (random sample")
        ("RANSAC.THRESH", po::value<int>(&RANSAC_THRESH) -> default_value(100*translation.thresh), "consensus) model approximation of the motions")
        ("RANSAC.EPS",    po::value<int>(&RANSAC_EPS)    -> default_value(100*translation.eps),    "size: kind of motion, thresh: max error to")
        ("RANSAC.PROB",   po::value<int>(&RANSAC_PROB)   -> default_value(100*translation.prob),   "classify as inlier, eps: max outliers ratio, prob: probability of success")

        ("HOUGH.MIN_DIST",        po::value<int>(&MIN_DIST)        -> default_value(50),  "Minimum distance between detected centers")
        ("HOUGH.MIN_RADIUS",      po::value<int>(&MIN_RADIUS)      -> default_value(15),  "Minimum/Maximum radius to be detected. ")
        ("HOUGH.MAX_RADIUS",      po::value<int>(&MAX_RADIUS)      -> default_value(25),  "If =0 every radius is possible")
        ("HOUGH.CANNY_THRESH",    po::value<int>(&CANNY_THRESH)    -> default_value(100), "Upper threshold for the Canny edge detector")
        ("HOUGH.CENTER_THRESH",   po::value<int>(&CENTER_THRESH)   -> default_value(8),   "Threshold for center detection")
        ("HOUGH.ACCUMULATOR_RES", po::value<int>(&ACCUMULATOR_RES) -> default_value(1),   "The inverse ratio of resolution\n")

        ("MOG.LEARNING", po::value<int>(&MOG_LEARNINGRATE) -> default_value(10), "Learning rate of all the background substractor classes")
    ;
}

void OnClick(int event, int x, int y, int, void* frame) { if ( event == EVENT_LBUTTONDOWN ) RESET = 1; }

// main ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
int main(int argc, char *argv[]) {
    cout <<"\033[00;30m";
    // initModule_nonfree();

    string videoName, outputName, configName, fourcc_code;

    po::options_description desc("Usage");
    desc.add_options()
        ("help,h",                                                                             "display this message")
        ("help-config",                                                                        "display the configuration parameters")
        ("onlineconfig,c",                                                                     "open configuration window")
        ("video,v",        po::value<string>(&videoName),                                      "use a video file as input\n(default: CV_CAP_ANY)")
        ("configfile,f",   po::value<string>(&configName) -> default_value("../config00.ini"), "use other than the default config file")
        ("writeto,w",      po::value<string>(&outputName),                                     "write videostream to the specified file")
        ("fourcc",         po::value<string>(&fourcc_code),                                    "video codec for writing stream to file")
        ("show-only",                                                                         "do nothing but to show and possibly write the stream")
    ;

    // Commandline parser, pass arguments to these five options
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << "\033[00;30mThis Program was written at the\033[1m Chair for Experimental Physics V at the TU Dortmund\033[0m" << endl;
        cout << "Press ESC or q to exit the program" << endl << "\t" << desc << endl;
        exit(EXIT_SUCCESS);
    }

    bool output(false);
    bool config(vm.count("onlineconfig"));

    VideoCapture cam;
    if (vm.count("video")) {
        videoName = "../video1/frame_%03d.jpg";  // hardcoded path to videoName

        cam.open(videoName);  // Load video file or a sequence of images
    } else {
        cam.open(CV_CAP_ANY);
    }

    if (!cam.isOpened()) {
        cout << "capture is not open. EXIT" << endl;
        exit(EXIT_FAILURE);
    }

    // Get the first image from cam/video
    Mat whole;  // container for the 'whole' video frame
    cam >> whole;  // Create first frame from capture/video; without this the HEIGHT/WIDTH etc. will not be defined

    // get the format of the video/image stream
    CAM_HEIGHT = cam.get(CV_CAP_PROP_FRAME_HEIGHT);
    CAM_WIDTH  = cam.get(CV_CAP_PROP_FRAME_WIDTH);

    VideoWriter writer;
    bool show_only(false);
    if (vm.count("writeto")) {
        if (vm.count("fourcc")) {
            int fourcc(CV_FOURCC(fourcc_code.c_str()[0], fourcc_code.c_str()[1], fourcc_code.c_str()[2], fourcc_code.c_str()[3]));
            writer.open(outputName, fourcc, 15, Size(CAM_WIDTH, CAM_HEIGHT));
        } else {
            writer.open(outputName, CV_FOURCC('F', 'F', 'V', '1'), 15, Size(CAM_WIDTH, CAM_HEIGHT));
        }

        if (writer.isOpened()) {
            cout << "VideoWriter is opened. Writing videostream to file " << outputName << endl;

            output = true;
        } else {
            cout << "VideoWriter couldn't be openend. EXIT" << endl;
            exit(EXIT_FAILURE);
        }

        show_only = vm.count("show-only");
    }


    // Configuration file parser
    po::options_description config_desc("Configuration");
    setConfig(config_desc);

    if (vm.count("help-config")) {
        cout << "\t" << config_desc << endl;
        exit(EXIT_SUCCESS);
    }

    po::variables_map config_vm;
    try {  // reading specified config file
        po::store(po::parse_config_file<char>(configName.c_str(), config_desc), config_vm);
        po::notify(config_vm);

        for (const std::pair<string, po::variable_value>& var : config_vm) {
            if (var.second.defaulted()) cout << "Parameter " << var.first << " was set to its default value! check config file" << endl;
        }
    }catch(po::reading_file& e) {  // file unreadable
        cout << "error: " << e.what() << "\nDefault values will be set." << endl;
        po::store(po::basic_parsed_options<char>(&config_desc), config_vm);
        po::notify(config_vm);
    }

    if (ROI_LEFT  > CAM_WIDTH*1./2)  ROI_LEFT  = static_cast<int>(CAM_WIDTH*1./2.1);  // Check for unvalid position of the roi
    if (ROI_RIGHT > CAM_WIDTH*1./2)  ROI_RIGHT = static_cast<int>(CAM_WIDTH*1./2.5);  // once these values are set to a valid state
    if (ROI_UPPER > CAM_HEIGHT*1./2) ROI_UPPER = static_cast<int>(CAM_HEIGHT*1./7.2);  // they will remain valid
    if (ROI_LOWER > CAM_HEIGHT*1./2) ROI_LOWER = static_cast<int>(CAM_HEIGHT*1./6);

    // H_MIN = 30;
    // V_MIN = S_MIN = 65;
    // H_MAX = V_MAX = S_MAX = 150;

    // B_MIN = G_MAX = 120;
    // G_MIN = R_MIN = 80;
    // B_MAX = R_MAX = 160;

    if (config) createconfigWindow();

    char key('a');
    int wait_ms(10);

    RESET = 0;
    ALARM = false;

    namedWindow("Video Frame");
    createTrackbar("waitKey duration:", "Video Frame", &wait_ms, 1000, nullptr);
    setMouseCallback("Video Frame", *OnClick);

    cout << "Video format is h: " << CAM_HEIGHT << ", w: " << CAM_WIDTH << "\nLoop starts..." << endl;

    while (key != 27 && key != 'q') {  // Create loop for streaming
        // end loop when 'ESC' or 'q' is pressed

        if (whole.empty()) {
            cam.open(videoName);
            cam >> whole;
            continue;
        }

        assert(RESET < 4);
        switch (RESET) {  // Ensure that at least two loops has RESET == true
            case 0: break;
            case 1: case 2: RESET++; break;
            case 3:  RESET = 0; ALARM = false; break;
        }

        if (!show_only) {
            GaussianBlur(whole, whole, Size(7, 7), 4);  // blurring reduces noise
            // medianBlur(binary, median, 5);

            stabilizeFrame(whole);  // stabile the current frame
            processFrame(whole);  // process the stabilized frame

            double fps = cam.get(CV_CAP_PROP_FPS);
            double pos = cam.get(CV_CAP_PROP_POS_FRAMES);

            Rect white(0, 0, 115, 60);
            rectangle(whole, white, Scalar::all(0), CV_FILLED, CV_AA, 0);

            stringstream s;
            s << "fps: " << fps;
            putText(whole, s.str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1, CV_AA);
            s.str("");
            s << "no: " << pos;
            putText(whole, s.str(), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1, CV_AA);
        }

        imshow("Video Frame", whole);  // show the results in windows
        setTrackbarPos("waitKey duration:", "Video Frame", wait_ms);

        key = waitKey(wait_ms);  // Capture Keyboard stroke

        cam >> whole;  // Get next frame; at the end of the loop because the first frame was retrieved before the loop started
        if (output) writer << whole;
    }
    cout << "Loop ended. Releasing capture and destroying windows..." << endl;

    cam.release();
    startWindowThread();  // Must be called to destroy the windows properly
    destroyAllWindows();  // otherwise the windows will remain openend.

    while (config && key != 'n' && key != 'N') {
        cout << "Do you want to store the new configration? (y/n) ";
        cin >> key;
        if (key == 'y' || key == 'Y') {
            cout << "Type the path of the new configration file: (type '-d' for default file '../config00.ini')" << endl;
            cin >> configName;
            if (configName == string("-d"))
                configName = "../config00.ini";
            cout << configName << endl;
            writeconfig(configName);
            key = 'n';
        }
    }

    exit(EXIT_SUCCESS);
}
// function declarations – functions used for actual processing ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void findFeatures(const Mat&, string method, vector<Point2f>&);
void clearFeaturesNotFound(vector<Point2f>&, vector<Point2f>&, vector<uchar>&, vector<float>&);
// void eraseLong(float);
void stabilize(int method, const vector<Point2f>&, const vector<Point2f>&, Mat&);
void morphologicalOperations(Mat&, int = 3);
void observePrimaryFiber(Mat&, Rect);
void drawCircles(Mat&, const vector<Vec3f>&, Scalar = Scalar(0, 0, 255));
void drawContours(Mat&);
void drawPoints(Mat&, const vector<Point2f>&);
void drawArrows(Mat&, const vector<Point2f>&, const vector<Point2f>&);
Point2f getOpticalFlow(const vector<Point2f>&, const vector<Point2f>&);
Point2f getDenseOpticalFlowVoting(const Mat&);
void showOpticalFlow(Mat&, const vector<Point2f>&, const vector<Point2f>&);
void convertDenseOpticalFlow(const Mat&, Mat&);
void deleteOutliers(vector<Vec3f>&);
void updateHistogram(const vector<Vec3f>&);
bool updatePrimaryWindow(const Vec3f&);
void calcDistances(const vector<Vec3f>&, vector<float>&);  // calculates the differences of the centers of circles
void sortCircles(vector<Vec3f>& circles) {  // sort circles by their x-coordinate in ascending order
    std::sort(circles.begin(), circles.end(), [](Vec3f a, Vec3f b){ return a[0] < b[0]; });}
    void getRelevantFlow(const Mat&, const vector<Point2f>&, vector<Point2f>&);
    void resizeMatWidth(Mat& mat, int newWidth, Scalar scalar = Scalar::all(0)) {
        mat = mat.t();
        mat.resize(newWidth, scalar);
        mat = mat.t();
    }
template<typename T> void print(const vector<T>& vec) {for (const T& x : vec) cout << x << " ";}

// actual doing something with the image –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void processFrame(Mat& whole) {
    Mat frame, drawframe, gray, binary, median, gauss, edges;
    static bool update_primary(true);

    //––– define region of interest (ROI) with parameters set by trackbars ––––––––––––––––––––––––––––––––––––––––––
    Rect ROI(Point(CAM_WIDTH*1./2-ROI_LEFT, CAM_HEIGHT*1./2-ROI_UPPER), Point(CAM_WIDTH*1./2+ROI_RIGHT, CAM_HEIGHT*1./2+ROI_LOWER));
    drawframe = whole(ROI);  // extract ROI
    drawframe.copyTo(frame);  // split in two frames; one for processing and one for drawing and showing
    imshow("Region of interest", frame);

    cvtColor(frame, gray, CV_BGR2GRAY);  // compute grayscale image
    imshow("Grayscale image", gray);

    threshold(gray, binary, BW_THRESH, 255, THRESH_BINARY_INV);  // compute b/w img
    imshow("Binary image", binary);

    //––– Detect edges in the blurred binary image using Canny algorithm ––––––––––––––––––––––––––––––––––––––––––––
    Canny(binary, edges, CANNY_THRESH, 2*CANNY_THRESH, 5);
    imshow("Edges", edges);

    //––– Detect circles in the binary image using the Hough transformation –––––––––––––––––––––––––––––––––––––––––
    vector<Vec3f> circles;  // Vektor für die gefundenen Kreise
    HoughCircles(binary, circles, CV_HOUGH_GRADIENT, ACCUMULATOR_RES, MIN_DIST, 2*CANNY_THRESH, CENTER_THRESH, MIN_RADIUS, MAX_RADIUS);
    // drawCircles(drawframe, circles);

    sortCircles(circles);  // sorting according to x-coordinate
    // print<Vec3f>(circles);
    deleteOutliers(circles);
    if (circles.empty()) return;

    //––– Somehow update the Histogram containing the circle center differences
    updateHistogram(circles);
    Vec3f last_circ = circles.back();
    if (update_primary) update_primary = updatePrimaryWindow(last_circ);

    Rect primary_fiber(MEAN_X_LEFTOFPRIMARY, MEAN_Y_ABOVEPRIMARY, MEAN_X_RIGHTOFPRIMARY - MEAN_X_LEFTOFPRIMARY, MEAN_Y_BELOWPRIMARY - MEAN_Y_ABOVEPRIMARY);

    //––– Use background substraction for detection of large errors –––––––––––––––––––––––––––––––––––––––––––––––––
    if (!update_primary) observePrimaryFiber(whole, primary_fiber);

    if (RESET) update_primary = true;

    rectangle(whole, ROI, Scalar(0, 255, 0), 1, CV_AA, 0);  // draw processed region of interest on whole image

    // draw circles and primary fiber window
    drawCircles(drawframe, circles, Scalar(0, 255, 0));
    circle(drawframe, Point(last_circ[0], last_circ[1]), last_circ[2], Scalar(255, 0, 0), 1, CV_AA, 0);  // draw circle
    circle(drawframe, Point(last_circ[0], last_circ[1]), 2, Scalar(255, 0, 0), -1, CV_AA, 0);      // draw center
    rectangle(whole, primary_fiber, Scalar(255, 0, 0), 1, CV_AA, 0);

    // pointing at the primary position of 'the blob' for testing purposes
    int x(515), y(205);
    line(whole, Point(x, y-30), Point(x, y+30), Scalar(255, 0, 0), 1, CV_AA);
    line(whole, Point(x-20, y), Point(x+20, y), Scalar(255, 0, 0), 1, CV_AA);
    ellipse(whole, RotatedRect(Point(x, y), Size(40, 60), 0), Scalar(255, 0, 0), 1, CV_AA);
}
void stabilizeFrame(Mat& whole) {
    static vector<Point2f> features_prev;
    vector<Point2f> features_curr;
    vector<uchar> status;
    vector<float> err;

    static Mat gray_prev;
    // static Mat stabframe_prev;
    Mat gray_curr;

    Rect ROI(Point(CAM_WIDTH*1./2-ROI_LEFT, CAM_HEIGHT*1./2-ROI_UPPER), Point(CAM_WIDTH*1./2+ROI_RIGHT, CAM_HEIGHT*1./2+ROI_LOWER));
    Mat stabframe = whole(ROI).clone();

    cvtColor(stabframe, gray_curr, CV_BGR2GRAY);  // compute grayscale image

    if (RESET) {
        gray_prev.release();
        features_prev.clear();
    }

    if (gray_prev.size() == gray_curr.size()) {
        //––– Sparse Optical Flow –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        // Feature matching using (sparse) optical flow of points using Lucas-Kanade-Method
        calcOpticalFlowPyrLK(gray_prev, gray_curr, features_prev, features_curr, status, err);
        clearFeaturesNotFound(features_prev, features_curr, status, err);  // clear features with status==false

        stabilize(RIGID_TRANSFORM, features_prev, features_curr, whole);

        //––– Dense Optical Flow ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        // Dense optical flow calculation using Farneback's method or the TVL1 method, no feature points involved
        // static Mat flow(gray_prev.size(), CV_32FC2);
        // void calcOpticalFlowFarneback(InputArray prev, InputArray next, InputOutputArray flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)
        // calcOpticalFlowFarneback(gray_prev, gray_curr, flow, 0.5, 5, 15, 3, 5, 1.1, OPTFLOW_USE_INITIAL_FLOW);
        // static Ptr<DenseOpticalFlow> optflowcalculator = createOptFlow_DualTVL1();
        // optflowcalculator->calc(gray_prev, gray_curr, flow);
        // calcOpticalFlowSF(stabframe_prev, stabframe, flow, 3, 2, 4);

        // medianBlur(flow, flow, 5);

        // Mat bgr(flow.size(), CV_32F);
        // convertDenseOpticalFlow(flow, bgr);
        // imshow("Dense optical flow", bgr);

        // getRelevantFlow(flow, features_prev, features_curr);
        // drawArrows(stabframe, features_curr, features_prev);
        // stabilize(VIDEOSTAB, features_prev, features_curr, whole);
    }

    // store for next loop
    gray_curr.copyTo(gray_prev);
    // stabframe.copyTo(stabframe_prev);
    findFeatures(gray_curr, "PyramidHARRIS", features_prev);  // find new "prev" features in current image and store

    drawPoints(stabframe, features_prev);
    imshow("Stabilizing frame", stabframe);
}
void findFeatures(const Mat& img, string method, vector<Point2f>& feat){  // find good tracking Features in frame 'img' (grayscale)
    vector<KeyPoint> keypoints;

    Ptr<FeatureDetector> detector = FeatureDetector::create(method);
    detector->detect(img, keypoints);

    KeyPoint::convert(keypoints, feat);  // store as Point2f

    // if (draw) drawKeypoints(drawframe, keypoints, drawframe, Scalar(0, 255, 255));
}
void clearFeaturesNotFound(vector<Point2f>& feat_prev, vector<Point2f>& feat_curr, vector<uchar>& stat, vector<float>& err) {  // only applicable if stat and feat_curr is defined (through calcOpticalFlowPyrLK)
    assert(feat_prev.size() == feat_curr.size() && feat_prev.size() == stat.size());  // and the two features sets are the same size

    auto iter_1 = feat_prev.begin();
    auto iter_2 = feat_curr.begin();
    auto found = stat.begin();
    auto it_err = err.begin();

    while (iter_1 < feat_prev.end()) {
        if (!(static_cast<bool>(*found))) {
            iter_1 = feat_prev.erase(iter_1);
            iter_2 = feat_curr.erase(iter_2);
            found = stat.erase(found);
            it_err = err.erase(it_err);
        } else {
            ++iter_1;
            ++iter_2;
            ++found;
            ++it_err;
        }
    }
}
void morphologicalOperations(Mat& in, int size) {
    // Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    Mat element = getGaussianKernel(size, -1);

    morphologyEx(in, in, MORPH_CLOSE, element);
    morphologyEx(in, in, MORPH_OPEN,  element);
}
void observePrimaryFiber(Mat& whole, Rect primary_fiber) {
    static Mat fgMOG, fgMOG2, fgGMG, fgMOGs, fgAll;
    static BackgroundSubtractorMOG  MOG;  // MOG approach
    static BackgroundSubtractorMOG2 MOG2(10, 16/*:default values*/, false/*=doShadowDetection*/);  // MOG2 approach
    static BackgroundSubtractorGMG  GMG;  // GMG approach
    // static bool ALERT_MOGS(false), ALERT_ALL(false);

    // Background substraction of the live image using MOG and MOG2
    Mat primary_fiber_window = whole(primary_fiber);

    MOG(primary_fiber_window, fgMOG, MOG_LEARNINGRATE*1./100);
    morphologicalOperations(fgMOG);
    MOG2(primary_fiber_window, fgMOG2, MOG_LEARNINGRATE*1./100);
    morphologicalOperations(fgMOG2);
    GMG(primary_fiber_window, fgGMG, MOG_LEARNINGRATE*1./100);
    morphologicalOperations(fgGMG, 5);

    multiply(fgMOG, fgMOG2, fgMOGs);
    multiply(fgGMG, fgMOGs, fgAll);
    morphologicalOperations(fgAll);

    int thresh_MOG(2000), thresh_MOG2(4000), thresh_GMG(20000), thresh_MOGs(2000), thresh_ALL(3000);
    int fill_MOG(1), fill_MOG2(1), fill_GMG(1), fill_MOGs(1), fill_All(1);
    Scalar color_MOG(0, 0, 255), color_MOG2(0, 0, 255), color_GMG(0, 0, 255), color_MOGs(0, 0, 255), color_All(0, 0, 255);

    if (norm(fgMOG)  > thresh_MOG)  { fill_MOG  = -1; color_MOG  = Scalar::all(0); }
    if (norm(fgMOG2) > thresh_MOG2) { fill_MOG2 = -1; color_MOG2 = Scalar::all(0); }
    if (norm(fgGMG)  > thresh_GMG)  { fill_GMG  = -1; color_GMG  = Scalar::all(0); }
    if (norm(fgMOGs) > thresh_MOGs) { fill_MOGs = -1; color_MOGs = Scalar::all(0); }
    if (norm(fgAll)  > thresh_ALL)  { fill_All  = -1; color_All  = Scalar::all(0); /*ALARM = true;*/ }

    circle(whole, Point(CAM_WIDTH-205, 25), 20, Scalar(0, 0, 255), fill_MOG, CV_AA, 0);
    putText(whole, "MOG", Point(CAM_WIDTH-219, 28), FONT_HERSHEY_SIMPLEX, 0.4, color_MOG, 1, CV_AA);
    circle(whole, Point(CAM_WIDTH-160, 25), 20, Scalar(0, 0, 255), fill_MOG2, CV_AA, 0);
    putText(whole, "MOG2", Point(CAM_WIDTH-177, 28), FONT_HERSHEY_SIMPLEX, 0.4, color_MOG2, 1, CV_AA);
    circle(whole, Point(CAM_WIDTH-115, 25), 20, Scalar(0, 0, 255), fill_GMG, CV_AA, 0);
    putText(whole, "GMG", Point(CAM_WIDTH-128, 28), FONT_HERSHEY_SIMPLEX, 0.4, color_GMG, 1, CV_AA);
    circle(whole, Point(CAM_WIDTH-70, 25), 20, Scalar(0, 0, 255), fill_MOGs, CV_AA, 0);
    putText(whole, "MOGs", Point(CAM_WIDTH-86, 28), FONT_HERSHEY_SIMPLEX, 0.4, color_MOGs, 1, CV_AA);
    circle(whole, Point(CAM_WIDTH-25, 25), 20, Scalar(0, 0, 255), fill_All, CV_AA, 0);
    putText(whole, "ALL", Point(CAM_WIDTH-35, 28), FONT_HERSHEY_SIMPLEX, 0.4, color_All, 1, CV_AA);

    stringstream s;
    s.str(""); s << "n:" << norm(fgMOG);
    fgMOG.resize(primary_fiber_window.size().height + 30, Scalar::all(255));
    putText(fgMOG, s.str(), Point(5, primary_fiber_window.size().height + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar::all(0), 1, CV_AA);
    s.str(""); s << "n:" << norm(fgMOG2);
    fgMOG2.resize(primary_fiber_window.size().height + 30, Scalar::all(255));
    putText(fgMOG2, s.str(), Point(5, primary_fiber_window.size().height + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar::all(0), 1, CV_AA);
    s.str(""); s << "n:" << norm(fgGMG);
    fgGMG.resize(primary_fiber_window.size().height + 30, Scalar::all(255));
    putText(fgGMG, s.str(), Point(5, primary_fiber_window.size().height + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar::all(0), 1, CV_AA);
    s.str(""); s << "n:" << norm(fgMOGs);
    fgMOGs.resize(primary_fiber_window.size().height + 30, Scalar::all(255));
    putText(fgMOGs, s.str(), Point(5, primary_fiber_window.size().height + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar::all(0), 1, CV_AA);
    s.str(""); s << "n:" << norm(fgAll);
    fgAll.resize(primary_fiber_window.size().height + 30, Scalar::all(255));
    putText(fgAll, s.str(), Point(5, primary_fiber_window.size().height + 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar::all(0), 1, CV_AA);

    imshow("Foreground Mask MOG",  fgMOG);
    imshow("Foreground Mask MOG2", fgMOG2);
    imshow("Foreground Mask GMG",  fgGMG);
    imshow("FG multiplication MOG*MOG2", fgMOGs);
    imshow("FG multiplication All", fgAll);

    // Mat primary_fiber_window_hsv_thresh, primary_fiber_window_bgr_thresh;
    // cvtColor(primary_fiber_window, primary_fiber_window_hsv_thresh, CV_BGR2HSV);
    // inRange(primary_fiber_window_hsv_thresh, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), primary_fiber_window_hsv_thresh);
    // inRange(primary_fiber_window, Scalar(B_MIN, G_MIN, R_MIN), Scalar(B_MAX, G_MAX, R_MAX), primary_fiber_window_bgr_thresh);

    // morphologicalOperations(primary_fiber_window_bgr_thresh);
    // morphologicalOperations(primary_fiber_window_hsv_thresh);

    // vector<vector<Point>> contours;
    // findContours(primary_fiber_window_hsv_thresh.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    // std::sort(contours.begin(), contours.end(), [](vector<Point> a, vector<Point> b){ return contourArea(a) > contourArea(b); });
    // Rect largestBlob = boundingRect(contours.front());

    // int fill_BGR_thresh(1);
    // Scalar color_BGR_thresh(0, 0, 255);

    // if (largestBlob.x + largestBlob.width*1./2 > primary_fiber_window.size().width * 1./2)  { fill_BGR_thresh  = -1; color_BGR_thresh  = Scalar::all(0); }

    // circle(whole, Point(CAM_WIDTH-25, 70), 20, Scalar(0, 0, 255), fill_BGR_thresh, CV_AA, 0);
    // putText(whole, "HSV", Point(CAM_WIDTH-37, 74), FONT_HERSHEY_SIMPLEX, 0.4, color_BGR_thresh, 1, CV_AA);
    // rectangle(primary_fiber_window_hsv_thresh, largestBlob, Scalar::all(128), 1, CV_AA, 0);

    // imshow("BGR thresholded image", primary_fiber_window_bgr_thresh);
    // imshow("HSV thresholded image", primary_fiber_window_hsv_thresh);
}
void getRelevantFlow(const Mat& flow, vector<Point2f>& prev, vector<Point2f>& curr) {
    assert(!prev.empty() && !flow.empty());

    Mat xy[2];  // X,Y
    split(flow, xy);
    curr.clear();

    for (auto i = prev.begin(); i != prev.end();) {
        float x = (*i).x; float y = (*i).y;

        if (x > flow.rows || y > flow.cols) {
            i = prev.erase(i);
        } else {
            float new_x = x + xy[0].at<float>(x, y);
            float new_y = y + xy[0].at<float>(x, y);

            curr.push_back(Point2f(new_x, new_y));
            ++i;
        }
    }
}
void stabilize_OpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    Point shift = getOpticalFlow(feat_prev, feat_curr);

    Mat trans_mat = (Mat_<float>(2, 3) << 1, 0, -shift.x, 0, 1, -shift.y);  // Creates a translation matrix with the mean optical flow
    warpAffine(whole, whole, trans_mat, whole.size());  // translates the whole frame with the matrix
}
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

    Mat M = findHomography(feat_prev, feat_curr, CV_RANSAC);

    M.at<double>(2, 2) = 1;
    M.at<double>(2, 1) = M.at<double>(2, 0) = 0;
    M.at<double>(0, 0) = M.at<double>(1, 1) = 1;
    M.at<double>(1, 0) = M.at<double>(0, 1) = 0;

    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);
}
// void stabilize_PhaseCorrelation(Mat& whole) {
//     Point2d shift = phaseCorrelate(gray_first, gray);
//     cout << shift << endl;

//     Mat trans_mat = (Mat_<float>(2,3) << 1, 0, shift.x, 0, 1, shift.y);  // Creates a translation matrix with the mean optical flow
//     warpAffine(whole, whole, trans_mat, whole.size(), WARP_INVERSE_MAP);  // translates the whole frame with the matrix
// }
void stabilize_Videostab(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    // static Ptr<OnePassStabilizer> onePassStabilizer = new OnePassStabilizer();

    static Mat prevMotions = Mat::eye(3, 3, CV_32FC1);

    if (RESET) prevMotions = Mat::eye(3, 3, CV_64F);

    // static vector<Mat> motions;  // here the motion prev->curr must be estimated
    // static int k = 1;

    Mat M = cv::videostab::estimateGlobalMotionRobust(feat_prev, feat_curr, cv::videostab::TRANSLATION, cv::videostab::RansacParams(RANSAC_SIZE, RANSAC_THRESH*1./100, RANSAC_EPS*1./100, RANSAC_PROB*1./100));
    prevMotions = M * prevMotions;

    warpPerspective(whole, whole, prevMotions, whole.size(), WARP_INVERSE_MAP);

    // k++;
}
void stabilize(int method, const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr, Mat& whole) {
    switch (method) {
        case PYR_LK_OPTICALFLOW: stabilize_OpticalFlow(feat_prev, feat_curr, whole);    break;
        case RIGID_TRANSFORM:    stabilize_RigidTransform(feat_prev, feat_curr, whole); break;
        case FIND_HOMOGRAPHY:    stabilize_Homography(feat_prev, feat_curr, whole);     break;
        case VIDEOSTAB:          stabilize_Videostab(feat_prev, feat_curr, whole);      break;
        // case PHASE_CORRELATION:  stabilize_PhaseCorrelation(whole);                     break;
        default: cout << "Chosen stationizing method no. " << method << " not associated with an implementation." << endl;
    }
}
Point2f getOpticalFlow(const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr) {  // calculate mean differences of the feature points
    assert(feat_prev.size() == feat_curr.size());

    // Averaging approach
    float x_diff_mean(0), y_diff_mean(0);
    int n(0);

    // Voting approach
    // Size fsize = frame.size();
    // TH2D votingHist("votingHist", "", fsize.width*2./VOTING_BINSIZE, -fsize.width, fsize.width,
                                      // fsize.height*2./VOTING_BINSIZE, -fsize.height, fsize.height);

    for (unsigned int i = 0; i < feat_prev.size(); i++) {
        // If Pyramidal Lucas Kanade didn't really find the feature, skip it
        // if (status[i] == false)  // removed due to the function 'clearFeaturesNotFound'
        //     continue;

        // Averaging approach
        n++;
        x_diff_mean += feat_curr[i].x - feat_prev[i].x;
        y_diff_mean += feat_curr[i].y - feat_prev[i].y;

        // Voting approach
        // float x_diff = feat_curr[i].x - feat_prev[i].x;
        // float y_diff = feat_curr[i].y - feat_prev[i].y;
        // votingHist.Fill(x_diff,y_diff);
    }

    // Averaging approach
    return Point2f(x_diff_mean/n, y_diff_mean/n);

    // int locmax, locmay, locmaz;
    // votingHist.GetMaximumBin(locmax, locmay, locmaz);
    // double max = votingHist.GetXaxis()->GetBinCenter(locmax);
    // double may = votingHist.GetYaxis()->GetBinCenter(locmay);
    // Point2f maxLoc(max, may);
    // cout << "Maximum Votes at " << maxLoc  << " contains " << votingHist.GetMaximum()*100./feat_prev.size() << "% of the found keypoints" << endl;
    // return maxLoc;
}
Point2f getDenseOpticalFlowVoting(const Mat& flow) {
    double VOTING_BINSIZE = 0.1;
    // Voting approach
    Size fsize(CAM_HEIGHT, CAM_WIDTH);
    TH2D votingHist("votingHist", "", fsize.width*2./VOTING_BINSIZE, -fsize.width, fsize.width,
      fsize.height*2./VOTING_BINSIZE, -fsize.height, fsize.height);

    Mat xy[2];  // X,Y
    split(flow, xy);

    for (int i = 0; i < flow.size().height; i++) {
        for (int j = 0; j < flow.size().width; j++) {
            double x_diff = xy[0].at<float>(i, j);
            double y_diff = xy[1].at<float>(i, j);
            if (x_diff == 0 && y_diff == 0) continue;
            votingHist.Fill(x_diff, y_diff);
        }
    }

    int locmax, locmay, locmaz;
    votingHist.GetMaximumBin(locmax, locmay, locmaz);
    double max = votingHist.GetXaxis()->GetBinCenter(locmax);
    double may = votingHist.GetYaxis()->GetBinCenter(locmay);
    Point2f maxLoc(max, may);
    // cout << "Maximum Votes at " << maxLoc  << " contains " << votingHist.GetMaximum()*100./feat_prev.size() << "% of the found keypoints" << endl;
    cout << "Maximum Votes at " << maxLoc  << " contains " << votingHist.GetMaximum()*100./flow.size().area() << "% of the flow points" << endl;
    return maxLoc;
}
template<typename T, int size>
vector<T> extractValues(const vector<Vec<T, size>> vec, int i) {
    vector<T> returnvec;
    for (const Vec<T, size>& x : vec) returnvec.push_back(x[i]);
        return returnvec;
}
void removeOuterOverflows(vector<Vec3f>& vec, vector<int>& indices) {
    int size(vec.size()-2);
    bool changed(true);

    while (changed) {
        changed = false;
        if (std::find(indices.begin(), indices.end(), size) != indices.end()) {
            vec.pop_back();
            indices.erase(std::find(indices.begin(), indices.end(), size));
            changed = true;
            // cout << "last element erased. " << endl;
        }
        if (std::find(indices.begin(), indices.end(), 0) != indices.end()) {
            vec.erase(vec.begin());
            indices.erase(std::find(indices.begin(), indices.end(), 0));
            for (uint j = 0; j < indices.size(); j++) indices[j]--;
                changed = true;
            // cout << "first element erased. " << endl;
        }
    }
}
bool updatePrimaryWindow(const Vec3f& last_circ) {
    static int n(0);

    if (RESET) n = 0;

    float x_leftofprimary  = last_circ[0] + MEAN_FIBER_DISTANCE * 1./2 + CAM_WIDTH*1./2 - ROI_LEFT;
    float x_rightofprimary = last_circ[0] + MEAN_FIBER_DISTANCE * 5./2 + CAM_WIDTH*1./2 - ROI_LEFT;
    float y_aboveprimary = last_circ[1] - MEAN_FIBER_DIAMETER * 5./2 + CAM_HEIGHT*1./2 - ROI_UPPER;
    float y_belowprimary = last_circ[1] + MEAN_FIBER_DIAMETER + CAM_HEIGHT*1./2 - ROI_UPPER;

    MEAN_X_LEFTOFPRIMARY = n*MEAN_X_LEFTOFPRIMARY + x_leftofprimary;
    MEAN_X_RIGHTOFPRIMARY = n*MEAN_X_RIGHTOFPRIMARY + x_rightofprimary;
    MEAN_Y_ABOVEPRIMARY = n*MEAN_Y_ABOVEPRIMARY + y_aboveprimary;
    MEAN_Y_BELOWPRIMARY = n*MEAN_Y_BELOWPRIMARY + y_belowprimary;

    n++;

    MEAN_X_LEFTOFPRIMARY /= n;
    MEAN_X_RIGHTOFPRIMARY /= n;
    MEAN_Y_ABOVEPRIMARY /= n;
    MEAN_Y_BELOWPRIMARY /= n;

    return !(n > 50);
}
void updateHistogram(const vector<Vec3f>& circles) {
    if (circles.empty()) return;
    static TH1D fiberDistances("fiberDistances", "Distances of Fiber Threads", 100, MIN_DIST, 150);
    static TH1D fiberDiameters("fiberDiameters", "Diameter of Fiber Threads", 100, 0, 2*MAX_RADIUS);

    vector<float> circle_distances;
    calcDistances(circles, circle_distances);

    // cout << "Center Distances: [";
    // print<float>(circle_distances);
    // cout << "]" << endl;

    // vector<int> overflows;
    for (uint i = 0; i < circle_distances.size(); i++) {
        /*int bin = */fiberDistances.Fill(circle_distances[i]);
        // if (bin <= 0 || bin > fiberDistances.GetNbinsX()) overflows.push_back(i);
    }

    // removeOuterOverflows(circles, overflows);

    for (const Vec3f& x : circles) fiberDiameters.Fill(2*x[2]);

    MEAN_FIBER_DISTANCE = fiberDistances.GetMean();  // update the global mean
    MEAN_FIBER_DIAMETER = fiberDiameters.GetMean();  // update the global mean

    // cout << "Mean fiber distances: " << MEAN_FIBER_DISTANCE << "\tEntries: " << fiberDistances.GetEntries();
    // cout << "\tMean fiber diameter: " << MEAN_FIBER_DIAMETER << "\tEntries: " << fiberDiameters.GetEntries() << endl;
}
void calcDistances(const vector<Vec3f>& circles, vector<float>& out) {
// calculate the differences of the circle centers assuming the circles are ordered ascending according to their x-coordinate
    out.clear();
    if (circles.empty()) return;

    Vec3f last = circles.front();

    for (uint i=1; i < circles.size(); i++) {
        Vec3f c = circles[i];

        float diff = sqrt(pow(c[0] - last[0], 2) + pow(c[1] - last[1], 2));
        out.push_back(diff);
        last = c;
    }
}
void deleteOutliers(vector<Vec3f>& circles) {
// approach to delete the outlier circles, that are circles not corresponding to a fiber thread, assuming that most
// of the circles are corresponding to a thread, using 'least squares' for just the y-coordinates (sorted in x)

    if (circles.empty()) return;
    vector<float> y_coordinates = extractValues<float, 3>(circles, 1);
    vector<float> x_coordinates = extractValues<float, 3>(circles, 0);

    // stuff for computing the correct matrix size
    int max_x = cvRound(circles.back()[0]);
    int max_y = cvRound(*max_element(y_coordinates.begin(), y_coordinates.end()));
    Size matsize(max_x+2*MIN_RADIUS, max_y+2*MIN_RADIUS);
    static Mat drawframe(matsize, CV_8UC3);

    if (matsize.height > drawframe.size().height) drawframe.resize(matsize.height, 0);
    if (matsize.width > drawframe.size().width) resizeMatWidth(drawframe, matsize.width, 0);
    drawframe *= 0;

    // calculation of the means
    float y_coord_mean = 0, x_coord_mean = 0;
    for (float& y : y_coordinates) y_coord_mean += y;
        for (float& x : x_coordinates) x_coord_mean += x;
            y_coord_mean /= y_coordinates.size();
        x_coord_mean /= x_coordinates.size();

        int threshold_y = 7./10*MIN_RADIUS;
        int threshold_x = (NUM_OF_THREADS + 1) * MAX_RADIUS;

        line(drawframe, Point(0, y_coord_mean), Point(drawframe.cols, y_coord_mean), Scalar::all(128));

        line(drawframe, Point(x_coord_mean, 0), Point(x_coord_mean, drawframe.rows), Scalar::all(128));
        Rect inlier_rect(x_coord_mean - threshold_x, y_coord_mean - threshold_y, 2*threshold_x, 2*threshold_y);
        rectangle(drawframe, inlier_rect, Scalar::all(128), 1, CV_AA);

    // Results of the procedure, show rejected circles and inliners on the frame and remove outliers
        drawCircles(drawframe, circles);
        circles.erase(remove_if(circles.begin(), circles.end(),
            [inlier_rect](Vec3f a){
                return !inlier_rect.contains(Point(a[0], a[1]));
            })
        , circles.end());
        drawCircles(drawframe, circles, Scalar(0, 255, 0));

        imshow("Circle outlier detection", drawframe);
    }
// a bunch of drawing functions ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void drawCircles(Mat& drawframe, const vector<Vec3f>& circles, Scalar color) {
    for (const Vec3f& circ : circles) {
        // speichert die ersten beiden Einträge von vector<Vec3f> circles in center und den dritten Wert in radius
            Point center(circ[0], circ[1]);
            int radius(circ[2]);

        circle(drawframe, center, radius, color, 1, CV_AA, 0);  // draw circle
        circle(drawframe, center, 2, color, -1, CV_AA, 0);      // draw center
    }
}
void drawPoints(Mat& drawframe, const vector<Point2f>& feat) {
    if (feat.empty()) cout << "Vector of Point2f \"feat\" empty." << endl;
    for (const Point2f& x : feat) {
        circle(drawframe, x, 2, Scalar(0, 255, 255), -1, CV_AA, 0);
    }
}
void drawArrows(Mat& drawframe, const vector<Point2f>& feat_1, const vector<Point2f>& feat_2) {
    assert(feat_1.size() == feat_2.size());  // zeichne Pfeile zwischen den zugeordneten Paaren von Punkten

    for (unsigned int i = 0; i < feat_1.size(); i++) {
        arrowedLine(drawframe, feat_1[i], feat_2[i], Scalar(255, 0, 0), 1, CV_AA, 0, 0.3);
    }
}
void showOpticalFlow(Mat& drawframe, const vector<Point2f>& feat_prev, const vector<Point2f>& feat_curr) {
    Point shift = getOpticalFlow(feat_prev, feat_curr);
    Rect white(0, 0, 200, 60);
    rectangle(drawframe, white, Scalar::all(0), CV_FILLED, CV_AA, 0);

    stringstream s;
    s.str("");
    s << "x_shift: " << shift.x;
    putText(drawframe, s.str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1, CV_AA);
    s.str("");
    s << "y_shift: " << shift.y;
    putText(drawframe, s.str(), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255), 1, CV_AA);
}
void convertDenseOpticalFlow(const Mat& flow, Mat& out) {
    Mat xy[2];  // X,Y
    split(flow, xy);

    // calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    // translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0/mag_max);

    // build hsv image
    Mat _hls[3], hsv;
    _hls[0] = angle;
    _hls[1] = Mat::ones(angle.size(), CV_32F);
    _hls[2] = magnitude;
    merge(_hls, 3, hsv);

    cvtColor(hsv, out, COLOR_HSV2BGR);
}
// configuration functions –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void correctValues(int, void*) {  // setting correct (>0) value for some parameters
    if (CANNY_THRESH == 0)
        CANNY_THRESH++;
    if (CENTER_THRESH == 0)
        CENTER_THRESH++;
    if (ACCUMULATOR_RES == 0)
        ACCUMULATOR_RES++;
    if (ROI_LOWER < 10)
        ROI_LOWER = 10;
    if (ROI_UPPER < 10)
        ROI_UPPER = 10;
    if (ROI_LEFT < 10)
        ROI_LEFT = 10;
    if (ROI_RIGHT < 10)
        ROI_RIGHT = 10;
    // if (VOTING_BINSIZE == 0)
        // VOTING_BINSIZE++;
    if (RANSAC_SIZE == 0)
        RANSAC_SIZE++;
    if (RANSAC_EPS == 0)
        RANSAC_EPS++;
    if (RANSAC_THRESH == 0)
        RANSAC_THRESH++;
    if (RANSAC_PROB == 0)
        RANSAC_PROB++;
    if (MIN_DIST < 15)  // not necessarily, but a smaller value slows down the program
        MIN_DIST = 15;

    // update the trackbar positions
    setTrackbarPos("Center threshhold:", "configration window", CENTER_THRESH);
    setTrackbarPos("Canny lower threshhold:", "configration window", CANNY_THRESH);
    setTrackbarPos("Reciprocal accumulator resolution:", "configration window", ACCUMULATOR_RES);
    setTrackbarPos("Lower boundary of ROI:", "configration window", ROI_LOWER);
    setTrackbarPos("Upper boundary of ROI:", "configration window", ROI_UPPER);
    setTrackbarPos("Left boundary of ROI: ",  "configration window", ROI_LEFT);
    setTrackbarPos("Right boundary of ROI:", "configration window", ROI_RIGHT);
    setTrackbarPos("Minimum distance between circles:", "configration window", MIN_DIST);

    setTrackbarPos("Subset size:", "RANSAC Parameters", RANSAC_SIZE);
    setTrackbarPos("Error threshold (%):", "RANSAC Parameters", RANSAC_THRESH);
    setTrackbarPos("Maximum outlier ratio (%):", "RANSAC Parameters", RANSAC_EPS);
    setTrackbarPos("Success probability (%):", "RANSAC Parameters", RANSAC_PROB);
}
void min_callback(int, void*) {  // min/max_callback functions preserve min<=max
    if (MAX_RADIUS < MIN_RADIUS) MAX_RADIUS = MIN_RADIUS;
    setTrackbarPos("Maximum radius of circles:", "configration window", MAX_RADIUS);
}
void max_callback(int, void*) {
    if (MAX_RADIUS < MIN_RADIUS) MIN_RADIUS = MAX_RADIUS;
    setTrackbarPos("Minimum radius of circles:", "configration window", MIN_RADIUS);
}
// functions opening and updating windows ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void createconfigWindow() {  // create the window where the configration trackbars should be displayed (config==true)
    namedWindow("configration window" , WINDOW_NORMAL);
    namedWindow("RANSAC Parameters", WINDOW_NORMAL);
    // namedWindow("HSV/BGR Thresholds", WINDOW_NORMAL);

    createTrackbar("Lower boundary of ROI:", "configration window", &ROI_LOWER, CAM_HEIGHT*1./2, correctValues);
    createTrackbar("Upper boundary of ROI:", "configration window", &ROI_UPPER, CAM_HEIGHT*1./2, correctValues);
    createTrackbar("Left boundary of ROI: ", "configration window", &ROI_LEFT,  CAM_WIDTH*1./2,  correctValues);
    createTrackbar("Right boundary of ROI:", "configration window", &ROI_RIGHT, CAM_WIDTH*1./2,  correctValues);

    createTrackbar("B/W threshhold:", "configration window", &BW_THRESH, 255, nullptr);

    createTrackbar("Canny lower threshhold:", "configration window", &CANNY_THRESH, 1000, correctValues);
    createTrackbar("Center threshhold:", "configration window", &CENTER_THRESH, 100, correctValues);
    createTrackbar("Reciprocal accumulator resolution:", "configration window", &ACCUMULATOR_RES, 5, correctValues);
    createTrackbar("Minimum distance between circles:", "configration window", &MIN_DIST, 255, correctValues);
    createTrackbar("Minimum radius of circles:", "configration window", &MIN_RADIUS, 100, min_callback);
    createTrackbar("Maximum radius of circles:", "configration window", &MAX_RADIUS, 100, max_callback);

    // createTrackbar("Binsize of Voting Histogram:", "configration window", &VOTING_BINSIZE, 100, correctValues);
    createTrackbar("MOG1/2 Learning rate (%): ", "configration window", &MOG_LEARNINGRATE, 99, nullptr);

    createTrackbar("Subset size:",     "RANSAC Parameters", &RANSAC_SIZE, 10, correctValues);
    createTrackbar("Error threshold (%):", "RANSAC Parameters", &RANSAC_THRESH, 99, correctValues);
    createTrackbar("Maximum outlier ratio (%):", "RANSAC Parameters", &RANSAC_EPS, 99, correctValues);
    createTrackbar("Success probability (%):",   "RANSAC Parameters", &RANSAC_PROB, 99, correctValues);

/*    createTrackbar("H Min:", "HSV/BGR Thresholds", &H_MIN, 255, nullptr);
    createTrackbar("H Max:", "HSV/BGR Thresholds", &H_MAX, 255, nullptr);
    createTrackbar("V Min:", "HSV/BGR Thresholds", &V_MIN, 255, nullptr);
    createTrackbar("V Max:", "HSV/BGR Thresholds", &V_MAX, 255, nullptr);
    createTrackbar("S Min:", "HSV/BGR Thresholds", &S_MIN, 255, nullptr);
    createTrackbar("S Max:", "HSV/BGR Thresholds", &S_MAX, 255, nullptr);

    createTrackbar("B Min:", "HSV/BGR Thresholds", &B_MIN, 255, nullptr);
    createTrackbar("B Max:", "HSV/BGR Thresholds", &B_MAX, 255, nullptr);
    createTrackbar("G Min:", "HSV/BGR Thresholds", &G_MIN, 255, nullptr);
    createTrackbar("G Max:", "HSV/BGR Thresholds", &G_MAX, 255, nullptr);
    createTrackbar("R Min:", "HSV/BGR Thresholds", &R_MIN, 255, nullptr);
    createTrackbar("R Max:", "HSV/BGR Thresholds", &R_MAX, 255, nullptr);*/
}
// writing to the configration files –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
void writeconfig(string configName) {
    ofstream outputFile(configName);
    // cout << configName << endl;

    if (outputFile.is_open()) {
        outputFile << "BW_THRESH="      << BW_THRESH      << endl;
        outputFile << "NUM_OF_THREADS=" << NUM_OF_THREADS << endl;
        outputFile << "\n[ROI]"             << endl;
        outputFile << "UPPER=" << ROI_UPPER << endl;
        outputFile << "LOWER=" << ROI_LOWER << endl;
        outputFile << "LEFT="  << ROI_LEFT  << endl;
        outputFile << "RIGHT=" << ROI_RIGHT << endl;
        outputFile << "\n[HOUGH]"                           << endl;
        outputFile << "MIN_DIST="        << MIN_DIST        << endl;
        outputFile << "MIN_RADIUS="      << MIN_RADIUS      << endl;
        outputFile << "MAX_RADIUS="      << MAX_RADIUS      << endl;
        outputFile << "CANNY_THRESH="    << CANNY_THRESH    << endl;
        outputFile << "CENTER_THRESH="   << CENTER_THRESH   << endl;
        outputFile << "ACCUMULATOR_RES=" << ACCUMULATOR_RES << endl;
        outputFile << "\n[RANSAC]"               << endl;
        outputFile << "SIZE="   << RANSAC_SIZE   << endl;
        outputFile << "THRESH=" << RANSAC_THRESH << endl;
        outputFile << "EPS="    << RANSAC_EPS    << endl;
        outputFile << "PROB="   << RANSAC_PROB   << endl;
        outputFile << "\n[MOG]"                       << endl;
        outputFile << "LEARNING=" << MOG_LEARNINGRATE << endl;
        outputFile.close();

        cout << "Succesfully written configration parameters to file." << endl;
    } else {
        cout << "Unable to open output file." << endl;
    }
}
// deprecated functions ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
// void eraseLong(float threshold) {
//     assert (features_prev.size() == features_curr.size() && status.size() == features_curr.size());

//     auto iter_1 = features_prev.begin();
//     auto iter_2 = features_curr.begin();
//     auto found = status.begin();
//     auto it_err = err.begin();

//     //vector<Point2f>::iterator end = features_prev.end();
//     while (iter_1 < features_prev.end()) {

//         float x_diff = (*iter_2).x - (*iter_1).x;
//         float y_diff = (*iter_2).y - (*iter_1).y;
//         float c = pow((pow(x_diff, 2) + pow(y_diff, 2)), 0.5);

//         if (threshold < c) {
//             iter_1 = features_prev.erase(iter_1);
//             iter_2 = features_curr.erase(iter_2);
//             found = status.erase(found);
//             it_err = err.erase(it_err);
//         } else {
//             ++iter_1;
//             ++iter_2;
//             ++found;
//             ++it_err;
//         }
//     }
// }
// void readconfig() {
//     string name;
//     int value;

//     bool set_BW_THRESH(false), set_voting_binsize(false), set_MOG_LEARNING(false);

//     bool set_accumulator_res(false), set_min_dist(false), set_canny_thresh(false), set_center_thresh(false), set_min_radius(false), set_max_radius(false);
//     bool set_ROI_UPPER(false), set_roi_lower(false), set_roi_left(false), set_roi_right(false);
//     bool set_ransac_eps(false), set_ransac_prob(false), set_ransac_thresh(false), set_ransac_size(false);

//     ifstream inputFile(configName);

//     if (!inputFile.is_open()) {
//         cout << "Input configration file not readable. ";
//         setDefaults();
//         return;
//     }

//     while (!inputFile.eof()) {
//         inputFile >> name >> value;
//         // cout << name << ": " << value << endl;

//         if (name == string("BW_THRESH") && !set_BW_THRESH) {
//             BW_THRESH = value;
//             set_BW_THRESH = true;
//         }
//         if (name == string("CANNY_THRESH") && !set_canny_thresh) {
//             CANNY_THRESH = value;
//             set_canny_thresh = true;
//         }
//         if (name == string("CENTER_THRESH") && !set_center_thresh) {
//             CENTER_THRESH = value;
//             set_center_thresh = true;
//         }
//         if (name == string("MIN_DIST") && !set_min_dist) {
//             MIN_DIST = value;
//             set_min_dist = true;
//         }
//         if (name == string("MIN_RADIUS") && !set_min_radius) {
//             MIN_RADIUS = value;
//             set_min_radius = true;
//         }
//         if (name == string("MAX_RADIUS") && !set_max_radius) {
//             MAX_RADIUS = value;
//             set_max_radius = true;
//         }
//         if (name == string("ACCUMULATOR_RES") && !set_accumulator_res) {
//             ACCUMULATOR_RES = value;
//             set_accumulator_res = true;
//         }
//         if (name == string("ROI_UPPER") && !set_ROI_UPPER) {
//             ROI_UPPER = value;
//             set_ROI_UPPER = true;
//         }
//         if (name == string("ROI_LOWER") && !set_roi_lower) {
//             ROI_LOWER = value;
//             set_roi_lower = true;
//         }
//         if (name == string("ROI_LEFT") && !set_roi_left) {
//             ROI_LEFT = value;
//             set_roi_left = true;
//         }
//         if (name == string("ROI_RIGHT") && !set_roi_right) {
//             ROI_RIGHT = value;
//             set_roi_right = true;
//         }
//         if (name == string("VOTING_BINSIZE") && !set_voting_binsize) {
//             VOTING_BINSIZE = value;
//             set_voting_binsize = true;
//         }
//         if (name == string("RANSAC_SIZE") && !set_ransac_size) {
//             RANSAC_SIZE = value;
//             set_ransac_size = true;
//         }
//         if (name == string("RANSAC_EPS") && !set_ransac_eps) {
//             RANSAC_EPS = value;
//             set_ransac_eps = true;
//         }
//         if (name == string("RANSAC_PROB") && !set_ransac_prob) {
//             RANSAC_PROB = value;
//             set_ransac_prob = true;
//         }
//         if (name == string("RANSAC_THRESH") && !set_ransac_thresh) {
//             RANSAC_THRESH = value;
//             set_ransac_thresh = true;
//         }
//         if (name == string("MOG_LEARNING") && !set_MOG_LEARNING) {
//             MOG_LEARNINGRATE = value;
//             set_MOG_LEARNING = true;
//         }
//     }

//     inputFile.close();

//     bool hougH_SET(SET_ACCUMULATOR_RES && set_min_dist && set_canny_thresh && set_center_thresh && set_min_radius && set_max_radius);
//     bool roi_set(set_ROI_UPPER && set_roi_lower && set_roi_left && set_roi_right);
//     bool ransac_params_set(set_ransac_eps && set_ransac_prob && set_ransac_thresh && set_ransac_size);

//     bool all_set(set_BW_THRESH && set_voting_binsize && hougH_SET && roi_set && ransac_params_set && set_MOG_LEARNING);


//     if (all_set) {
//         cout << "Succesfully read configration parameters from file." << endl;
//     } else {
//         cout << "There was an error during the import of the configration parameters. ";
//         setDefaults();
//     }
// }
// void setDefaults() {
//     cout << "Setting default values for parameters." << endl;

//     ROI_UPPER = CAM_HEIGHT*1./7.2;
//     ROI_LOWER = CAM_HEIGHT*1./6;
//     ROI_LEFT  = CAM_WIDTH*1./2.1;
//     ROI_RIGHT = CAM_WIDTH*1./2.5;
//     BW_THRESH = 140;
//     CANNY_THRESH = 100;
//     CENTER_THRESH = 8;
//     ACCUMULATOR_RES = 1;
//     MIN_DIST = 50;
//     MIN_RADIUS = 15;
//     MAX_RADIUS = 25;
//     VOTING_BINSIZE = 5;
//     MOG_LEARNINGRATE = 10;

//     cv::videostab::RansacParams params = cv::videostab::RansacParams::translationMotionStd();  // standard parameter for translational motion model
//     RANSAC_SIZE = params.size,
//     RANSAC_THRESH = params.thresh*100;
//     RANSAC_EPS = params.eps*100;
//     RANSAC_PROB = params.prob*100;
// }
