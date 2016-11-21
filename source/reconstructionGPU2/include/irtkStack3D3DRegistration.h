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
#pragma once
#include "nDRegistration.h"

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <math.h>
#include <stdlib.h>

template <typename T>
class irtkStack3D3DRegistration : public nDRegistration<T>, public irtkObject
{
public:
  irtkStack3D3DRegistration(unsigned int target_idx, std::vector<irtkGenericImage<T> >* stacks, 
    std::vector<irtkRigidTransformation>* transformations, irtkGenericImage<char>* mask, std::vector<int> packageDenominator = std::vector<int>());
  ~irtkStack3D3DRegistration();

  virtual void run();
  std::vector<irtkRigidTransformation> getStackTransformations();

  //template <typename T> friend class ParallelStackRegistrations;

private:
  void ResetOrigin(irtkGreyImage image, irtkRigidTransformation transformation);
  void InvertStackTransformations(vector<irtkRigidTransformation>* stack_transformations);

  std::vector<irtkGenericImage<T> >* m_stacks;
  irtkGenericImage<char>* m_mask;
  unsigned int m_target_idx;
  std::vector<irtkRigidTransformation>* m_stack_transformations;
  std::vector<int> m_packageDenominator;

};