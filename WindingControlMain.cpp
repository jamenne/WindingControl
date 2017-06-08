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
#include "time.h"
#include <sstream>

//mvIMPACT stuff
#include <apps/Common/exampleHelper.h>
#include <common/minmax.h>
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>


using namespace std;

int main()
{	
	string input;

	DeviceManager devMgr;

	Device* pDev = OpenCam(devMgr);

	// cout << "Do you want to record a video? [y/n]" << endl;
	// cin >> input;

	// if((input == "y") | (input == "Yes") | (input == "yes") | (input == "Y")){

	// 	time_t sec = time(NULL);

	// 	tm *uhr = localtime(&sec);

	// 	stringstream path;
		
		
	// 	path << "/local/Wickel_" << uhr->tm_year-100 << uhr->tm_mon+1 << uhr->tm_mday << "-" << uhr->tm_hour << uhr->tm_min << uhr->tm_sec << ".avi";



	// 	WriteVideoToFile(path.str(), pDev);

	// }

	// if((input == "n") | (input == "No") | (input == "no") | (input == "N")){

	// 	cout << "Do you want to take a snapshot?" << endl;
	// 	cin >> input;
	// 	if((input == "y") | (input == "Yes") | (input == "yes") | (input == "Y")){

	// 		Mat frame;
	// 		stringstream path;

	// 		time_t sec = time(NULL);

	// 		tm *uhr = localtime(&sec);		

	// 		path << "/home/e5a-labor/pic/Wickel_" << uhr->tm_year-100 << uhr->tm_mon+1 << uhr->tm_mday << "-" << uhr->tm_hour << uhr->tm_min << uhr->tm_sec << ".png";

	// 		cout << path.str() << endl;


	// 		frame = ImageRequestSingle(pDev);

	// 		cout << "frame.size(): " << frame.size() << "mat.type(): " << frame.type() << endl;

	// 		/*vector<int> compression_params;
	// 		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	// 		compression_params.push_back(9);*/

	// 		try {
	// 			imwrite(path.str(), frame);
	// 			cout << "saved to " << path.str() << endl;
	// 		}
	// 		catch (runtime_error& ex) {
	// 			fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
	// 		}

	// 	}

	// 	else{

	// 		cout << "Camera picture will be just shown..." << endl;
	// 		ShowPictureOfCamera(pDev);

	// 	}


	// }
	
	// else{
	// 	cout << "Camera picture will be just shown..." << endl;
	// 	ShowPictureOfCamera(pDev);
	// }
	

	return 0;
}