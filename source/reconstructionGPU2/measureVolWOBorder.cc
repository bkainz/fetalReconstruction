/*=========================================================================

Library   : Image Registration Toolkit (IRTK)
Module    : $Id: convert.cc 968 2013-08-15 08:48:21Z kpk09 $
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2008 onwards
Date      : $Date: 2013-08-15 09:48:21 +0100 (Thu, 15 Aug 2013) $
Version   : $Revision: 968 $
Changes   : $Author: kpk09 $

=========================================================================*/

#include <irtkImage.h>
#include <string>
#include <fstream>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>

using namespace boost::filesystem;

void usage()
{
	cerr << "Usage: [in folder] \n\n";
	exit(1);
}

int main(int argc, char **argv)
{


	if (argc < 1) {
		usage();
	}


	std::vector<std::string> inputFiles;

	for ( boost::filesystem::recursive_directory_iterator end, dir(argv[1]); 
		dir != end; ++dir ) {

			path p = *dir;

			if (is_regular_file(p) )
			{
				if(p.extension() == path(".nii"))
				{
					cout << p << std::endl;   
					inputFiles.push_back(p.string());
				}
			}
	}


	ofstream evalFile;
	evalFile.open ("volumeMeasures.txt", ios::out | ios::app);

	for(int i =0 ; i < inputFiles.size(); i++)
	{
		string inname = inputFiles[i];
		irtkGreyImage img((char*)inname.c_str());
		irtkGreyImage new_img(img.GetImageAttributes());
		new_img = 0;


		unsigned int sum = 0;
		for(int z = 3; z < img.GetZ()-3; z++)
		{
			for(int y = 3; y < img.GetY()-3; y++)
			{
				for(int x = 3; x < img.GetX()-3; x++)
				{
					if(img(x,y,z) != 0)
						sum++;

					new_img(x,y,z) = img(x,y,z);
				}
			}
		}


		std::cout << img.GetXSize() << " " << img.GetYSize() << " " << img.GetZSize() << std::endl;
		sum *= (img.GetXSize()*img.GetYSize()*img.GetZSize());
		sum /= 1000;

		inname.erase(0, inname.find_last_of("\\")+1);

		string nname = string("c_") + inname;

		new_img.Write(nname.c_str());

		inname.erase(4,inname.size());
		std::cout << inname << " Volume = " << sum << " ml" << std::endl;
		evalFile << inname << " " << sum << std::endl;
	}

	evalFile.close();

	return EXIT_SUCCESS;
}
