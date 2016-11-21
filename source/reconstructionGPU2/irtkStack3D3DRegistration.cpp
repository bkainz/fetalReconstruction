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
#include "irtkStack3D3DRegistration.h"
#include <irtkImage.h>
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>

template <typename T>
irtkStack3D3DRegistration<T>::irtkStack3D3DRegistration(unsigned int target_idx, 
  std::vector<irtkGenericImage<T> >* stacks, std::vector<irtkRigidTransformation>* transformations, irtkGenericImage<char>* mask,
  std::vector<int> packageDenominator) :
  m_stacks(stacks), m_target_idx(target_idx), m_stack_transformations(transformations), m_mask(mask), m_packageDenominator(packageDenominator)
{

}

template <typename T>
irtkStack3D3DRegistration<T>::~irtkStack3D3DRegistration()
{

}


template <typename T>
class ParallelStackRegistrations {
  vector<irtkGenericImage<T> >& stacks;
  vector<irtkRigidTransformation>& stack_transformations;
  int templateNumber;
  irtkGreyImage& target;
  irtkRigidTransformation& offset;
  bool _externalTemplate;

public:
  ParallelStackRegistrations(vector<irtkGenericImage<T> >& _stacks,
    vector<irtkRigidTransformation>& _stack_transformations,
    int _templateNumber,
    irtkGreyImage& _target,
    irtkRigidTransformation& _offset,
    bool externalTemplate = false) :
    stacks(_stacks),
    stack_transformations(_stack_transformations),
    target(_target),
    offset(_offset) {
    templateNumber = _templateNumber,
      _externalTemplate = externalTemplate;
  }

  void operator() (const blocked_range<size_t> &r) const {
    for (size_t i = r.begin(); i != r.end(); ++i) {

      //do not perform registration for template
      if (i == templateNumber)
        continue;

      //rigid registration object
      irtkImageRigidRegistrationWithPadding registration;
      //irtkRigidTransformation transformation = stack_transformations[i];

      //set target and source (need to be converted to irtkGreyImage)
      irtkGreyImage source = stacks[i];

      //include offset in trasformation   
      irtkMatrix mo = offset.GetMatrix();
      irtkMatrix m = stack_transformations[i].GetMatrix();
      m = m*mo;
      stack_transformations[i].PutMatrix(m);

      //perform rigid registration
      registration.SetInput(&target, &source);
      registration.SetOutput(&stack_transformations[i]);
      if (_externalTemplate)
      {
        registration.GuessParameterThickSlicesNMI();
      }
      else
      {
        registration.GuessParameterThickSlices();
      }
      registration.SetTargetPadding(0);
      registration.Run();

      mo.Invert();
      m = stack_transformations[i].GetMatrix();
      m = m*mo;
      stack_transformations[i].PutMatrix(m);

      //stack_transformations[i] = transformation;            

      //save volumetric registrations
      if (false) {
        //buffer to create the name
        char buffer[256];
        registration.irtkImageRegistration::Write((char *) "parout-volume.rreg");
        sprintf(buffer, "stack-transformation%i.dof.gz", i);
        stack_transformations[i].irtkTransformation::Write(buffer);
        target.Write("target.nii.gz");
        sprintf(buffer, "stack%i.nii.gz", i);
        stacks[i].Write(buffer);
      }
    }
  }

  // execute
  void operator() () const {
    task_scheduler_init init(tbb_no_threads);
    parallel_for(blocked_range<size_t>(0, stacks.size()),
      *this);
    init.terminate();
  }

};

template <typename T>
void irtkStack3D3DRegistration<T>::InvertStackTransformations(vector<irtkRigidTransformation>* stack_transformations)
{
  //for each stack
  for (unsigned int i = 0; i < stack_transformations->size(); i++) {
    //invert transformation for the stacks
    stack_transformations->at(i).Invert();
    stack_transformations->at(i).UpdateParameter();
  }
}

template <typename T>
void irtkStack3D3DRegistration<T>::ResetOrigin(irtkGreyImage image, irtkRigidTransformation transformation)
{
  double ox, oy, oz;
  image.GetOrigin(ox, oy, oz);
  image.PutOrigin(0, 0, 0);
  transformation.PutTranslationX(ox);
  transformation.PutTranslationY(oy);
  transformation.PutTranslationZ(oz);
  transformation.PutRotationX(0);
  transformation.PutRotationY(0);
  transformation.PutRotationZ(0);
}

template <typename T>
void irtkStack3D3DRegistration<T>::run()
{
  cout << "Performing 3D-3D stack registration..." << endl;

  streambuf* strm_buffer = cout.rdbuf();
  streambuf* strm_buffer_e = cerr.rdbuf();
  ofstream file("log-3D3Dregistration.txt");
  ofstream file_e("log-3D3Dregistration-error.txt");
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());

  cout << "StackRegistrations" << endl;

  InvertStackTransformations(m_stack_transformations);

  //template is set as the target
  irtkGreyImage target;
  target = m_stacks->at(m_target_idx);

  //target needs to be masked before registration
  double x, y, z;
  for (int i = 0; i < target.GetX(); i++) {
    for (int j = 0; j < target.GetY(); j++) {
      for (int k = 0; k < target.GetZ(); k++) {
        //image coordinates of the target
        x = i;
        y = j;
        z = k;
        //change to world coordinates
        target.ImageToWorld(x, y, z);
        //change to mask image coordinates - mask is aligned with target
        m_mask->WorldToImage(x, y, z);
        x = round(x);
        y = round(y);
        z = round(z);
        //if the voxel is outside mask ROI set it to -1 (padding value)
        if ((x >= 0) && (x < m_mask->GetX()) && (y >= 0) && (y < m_mask->GetY()) && (z >= 0)
          && (z < m_mask->GetZ())) {
          if (m_mask->Get(x, y, z) == 0)
            target(i, j, k) = 0;
        }
        else
          target(i, j, k) = 0;
      }
    }
  }

  irtkRigidTransformation offset;
  ResetOrigin(target, offset);

  //register all stacks to the target
  ParallelStackRegistrations<T> registration(
    *m_stacks,
    *m_stack_transformations,
    m_target_idx,
    target,
    offset);
  registration();

  InvertStackTransformations(m_stack_transformations);

  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);
}

template <typename T>
std::vector<irtkRigidTransformation> irtkStack3D3DRegistration<T>::getStackTransformations()
{
  return *m_stack_transformations;
}

template class irtkStack3D3DRegistration < float >;
template class irtkStack3D3DRegistration < double >;
