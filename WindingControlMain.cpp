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
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>

using namespace std;

int main()
{	
	string input;

	DeviceManager devMgr;

	Device* pDev = OpenCam(devMgr);

	cout << "Do you want to record the image?" << endl;
	cin >> input;

	if((input == "y") | (input == "Yes") | (input == "yes")){

		time_t sec = time(NULL);

		tm *uhr = localtime(&sec);

		stringstream path;
		
		
		path << "Wickel_" << uhr->tm_year-100 << uhr->tm_mon+1 << uhr->tm_mday << "-" << uhr->tm_hour << uhr->tm_min << uhr->tm_sec << ".avi";



		WriteVideoToFile(path.str(), pDev);

	}

	
	else ShowPictureOfCamera(pDev);
	
	//bool stab=true;
   	//ProzessCapturedFrame("/home/e5a-labor/src/Winding Control/windingcontrol/build/Wickelvideos/Wickel_15101-103829.avi", stab);

	return 0;
}