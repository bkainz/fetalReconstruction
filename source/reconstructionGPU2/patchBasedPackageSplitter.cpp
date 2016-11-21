/*=========================================================================
* GPU accelerated motion compensation for MRI
*
* Copyright (c) 2016 Bernhard Kainz, Amir Alansary, Maria Kuklisova-Murgasova,
* Kevin Keraudren, Markus Steinberger
* (b.kainz@imperial.ac.uk)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
=========================================================================*/
#include "patchBasedPackageSplitter.h"
#include <irtkImage.h>
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>

template <typename T>
patchBasedPackageSplitter<T>::patchBasedPackageSplitter(unsigned int target_idx,
	std::vector<irtkGenericImage<T> >* stacks, std::vector<irtkRigidTransformation>* transformations, std::vector<T> thickness,
	std::vector<int> packageDenominator) :
	m_stacks(stacks), m_target_idx(target_idx), m_stack_transformations(transformations), m_packageDenominator(packageDenominator), m_stack_thickness(thickness)
{
	makePackageVolumes();
}

template <typename T>
patchBasedPackageSplitter<T>::~patchBasedPackageSplitter()
{

}

template <typename T>
std::vector<irtkGenericImage<T> > patchBasedPackageSplitter<T>::getPackageVolumes()
{

	return m_packages;
}

template <typename T>
std::vector<irtkRigidTransformation> patchBasedPackageSplitter<T>::getPackageTransformations()
{

	return m_packages_transformations;
}


template <typename T>
std::vector<T> patchBasedPackageSplitter<T>::getPackageThickness()
{

	return m_packages_thickness;
}




//TODO

template <typename T>
void patchBasedPackageSplitter<T>::makePackageVolumes()
{

	//TODO: fill m_packages so that each package contains the temporal correct substack
	//TODO: distribute external stack transformations correctly to m_stack_transformations
	for (unsigned int s = 0; s < m_stacks->size(); s++)
	{
		irtkGenericImage<T> image = m_stacks->at(s);
		int packages = m_packageDenominator[s];
		irtkImageAttributes attr = image.GetImageAttributes();

		//slices in package
		int pkg_z = attr._z / packages;
		double pkg_dz = attr._dz*packages;
		//cout << "packages: " << packages << "; slices: " << attr._z << "; slices in package: " << pkg_z << endl;
		//cout << "slice thickness " << attr._dz << "; slickess thickness in package: " << pkg_dz << endl;

		//char buffer[256];
		int i, j, k, l;
		double x, y, z, sx, sy, sz, ox, oy, oz;
		for (l = 0; l < packages; l++)
		{
			attr = image.GetImageAttributes();
			if ((pkg_z*packages + l) < attr._z)
				attr._z = pkg_z + 1;
			else
				attr._z = pkg_z;
			attr._dz = pkg_dz;

			//cout << "split image " << l << " has " << attr._z << " slices." << endl;

			//fill values in each stack
			irtkRealImage stack(attr);
			stack.GetOrigin(ox, oy, oz);

			//cout << "Stack " << l << ":" << endl;
			for (k = 0; k < stack.GetZ(); k++)
				for (j = 0; j < stack.GetY(); j++)
					for (i = 0; i < stack.GetX(); i++)
						stack.Put(i, j, k, image(i, j, k*packages + l));

			//adjust origin

			//original image coordinates
			x = 0; y = 0; z = l;
			image.ImageToWorld(x, y, z);
			//cout << "image: " << x << " " << y << " " << z << endl;
			//stack coordinates
			sx = 0; sy = 0; sz = 0;
			stack.PutOrigin(ox, oy, oz); //adjust to original value
			stack.ImageToWorld(sx, sy, sz);
			//cout << "stack: " << sx << " " << sy << " " << sz << endl;
			//adjust origin
			//cout << "adjustment needed: " << x - sx << " " << y - sy << " " << z - sz << endl;
			stack.PutOrigin(ox + (x - sx), oy + (y - sy), oz + (z - sz));
			sx = 0; sy = 0; sz = 0;
			stack.ImageToWorld(sx, sy, sz);
			//cout << "adjusted: " << sx << " " << sy << " " << sz << endl;

			//sprintf(buffer, "package%i_%i.nii.gz", s, l);
			//stack.Write(buffer);
			//stacks.push_back(stack)
			m_packages.push_back(stack);
			m_packages_transformations.push_back(m_stack_transformations->at(s));
			m_packages_thickness.push_back(m_stack_thickness[s]);
		}
	}
}


template class patchBasedPackageSplitter < float >;
template class patchBasedPackageSplitter < double >;