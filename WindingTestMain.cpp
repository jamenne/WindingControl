// WindingControlMain.cpp
//
// Copyright 2015 TU Dortmund, Physik, Experimentelle Physik 5
//
// Author: Janine Müller
//
//
//
#include "WindingTest.h"
#include <iostream>
#include "time.h"
#include <sstream>

using namespace std;

int main()
{	
	
	bool stab=true;
	bool ChooseROI=true;
   	ProzessCapturedFrame("/Users/janine/WindingControl/Wickelvideos/Wickel_15101-103829.avi", stab, ChooseROI);

	//TestCallBack();

	return 0;
}