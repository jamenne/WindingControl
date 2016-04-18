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

	/*cout << "Do you want to record the image?" << endl;
	cin >> input;

	if((input == "y") | (input == "Yes") | (input == "yes") | (input == "Y")){

		time_t sec = time(NULL);

		tm *uhr = localtime(&sec);

		stringstream path;
		
		
		path << "/local/Wickel_" << uhr->tm_year-100 << uhr->tm_mon+1 << uhr->tm_mday << "-" << uhr->tm_hour << uhr->tm_min << uhr->tm_sec << ".avi";



		WriteVideoToFile(path.str(), pDev);

	}

	
	else ShowPictureOfCamera(pDev);*/
	
	bool On=false;

	cout << "Do you want to switch the Output on?" << endl;
	cin >> input;

	if (input == "y")
	{
		On = true;
	}

	int number = 0;
	cout << "Which one do you want to switch on?" << endl;
	cin >> number;

	SetOutput(pDev, number, On );

	return 0;
}