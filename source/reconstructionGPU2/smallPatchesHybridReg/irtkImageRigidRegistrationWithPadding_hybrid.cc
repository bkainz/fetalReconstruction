/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageRigidRegistration.cc 510 2012-01-17 10:46:20Z mm3 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2012-01-17 10:46:20 +0000 (Tue, 17 Jan 2012) $
  Version   : $Revision: 510 $
  Changes   : $Author: mm3 $

=========================================================================*/

#include <irtkRegistration.h>

#include <irtkHomogeneousTransformationIterator.h>

#include <irtkImageRigidRegistrationWithPadding_hybrid.h>

#include <irtkMultiThreadedImageRigidRegistrationWithPadding.h>

#include <irtkGaussianBlurring.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

irtkImageRigidRegistrationWithPadding_hybrid::irtkImageRigidRegistrationWithPadding_hybrid() : irtkImageRegistrationWithPadding()
{
  ccCostFunction_gpu = new ccSimilarityMetric(0);
}

irtkImageRigidRegistrationWithPadding_hybrid::~irtkImageRigidRegistrationWithPadding_hybrid()
{

}


void irtkImageRigidRegistrationWithPadding_hybrid::GuessParameter()
{
  int i;
  double xsize, ysize, zsize;

  if ((_target == NULL) || (_source == NULL)) {
    cerr << "irtkImageRigidRegistrationWithPadding_hybrid::GuessParameter: Target and source image not found" << endl;
    exit(1);
  }

  // Default parameters for registration
  _NumberOfLevels     = 3;
  _NumberOfBins       = 64;

  // Default parameters for optimization
  _SimilarityMeasure  = NMI;
  _OptimizationMethod = GradientDescent;
  _Epsilon            = 0.0001;

  // Read target pixel size
  _target->GetPixelSize(&xsize, &ysize, &zsize);

  // Default target parameters
  _TargetBlurring[0]      = GuessResolution(xsize, ysize, zsize) / 2.0;
  _TargetResolution[0][0] = GuessResolution(xsize, ysize, zsize);
  _TargetResolution[0][1] = GuessResolution(xsize, ysize, zsize);
  _TargetResolution[0][2] = GuessResolution(xsize, ysize, zsize);

  for (i = 1; i < _NumberOfLevels; i++) {
    _TargetBlurring[i]      = _TargetBlurring[i-1] * 2;
    _TargetResolution[i][0] = _TargetResolution[i-1][0] * 2;
    _TargetResolution[i][1] = _TargetResolution[i-1][1] * 2;
    _TargetResolution[i][2] = _TargetResolution[i-1][2] * 2;
  }

  // Read source pixel size
  _source->GetPixelSize(&xsize, &ysize, &zsize);

  // Default source parameters
  _SourceBlurring[0]      = GuessResolution(xsize, ysize, zsize) / 2.0;
  _SourceResolution[0][0] = GuessResolution(xsize, ysize, zsize);
  _SourceResolution[0][1] = GuessResolution(xsize, ysize, zsize);
  _SourceResolution[0][2] = GuessResolution(xsize, ysize, zsize);

  for (i = 1; i < _NumberOfLevels; i++) {
    _SourceBlurring[i]      = _SourceBlurring[i-1] * 2;
    _SourceResolution[i][0] = _SourceResolution[i-1][0] * 2;
    _SourceResolution[i][1] = _SourceResolution[i-1][1] * 2;
    _SourceResolution[i][2] = _SourceResolution[i-1][2] * 2;
  }

  // Remaining parameters
  for (i = 0; i < _NumberOfLevels; i++) {
    _NumberOfIterations[i] = 20;
    _NumberOfSteps[i]      = 4;
    _LengthOfSteps[i]      = 2 * pow(2.0, i);
  }

  // Try to guess padding by looking at voxel values in all eight corners of the volume:
  // If all values are the same we assume that they correspond to the padding value
  _TargetPadding = MIN_GREY;
  if ((_target->Get(_target->GetX()-1, 0, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, 0, _target->GetZ()-1)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, 0)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, 0, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, _target->GetZ()-1) == _target->Get(0, 0, 0))) {
    _TargetPadding = _target->Get(0, 0, 0);
  }
  
  _SourcePadding = MIN_GREY;
  if ((_source->Get(_source->GetX()-1, 0, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, 0, _source->GetZ()-1)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, 0)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, 0, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, _source->GetZ()-1) == _source->Get(0, 0, 0))) {
    _SourcePadding = _source->Get(0, 0, 0);
  }
  cerr<<endl<<endl;
  cerr<<"GuessParameter:"<<endl;
  cerr<<"Target padding is "<<_TargetPadding<<endl;
  cerr<<"Source padding is "<<_SourcePadding<<endl;
  cerr<<endl<<endl;

}

void irtkImageRigidRegistrationWithPadding_hybrid::GuessParameterThickSlices()
{
  int i;
  double xsize, ysize, zsize,size;

  if ((_target == NULL) || (_source == NULL)) {
    cerr << "irtkImageRigidRegistrationWithPadding_hybrid::GuessParameter: Target and source image not found" << endl;
    exit(1);
  }

  // Default parameters for registration
  _NumberOfLevels     = 3;
  _NumberOfBins       = 64;

  // Default parameters for optimization
  _SimilarityMeasure  = CC;
  _OptimizationMethod = GradientDescent;
  _Epsilon            = 0.0001;

  // Read target pixel size
  _target->GetPixelSize(&xsize, &ysize, &zsize);
  
  if (ysize<xsize)
    size = ysize;
  else
    size = xsize;
  //if (zsize<size)
    //size = zsize;  

  // Default target parameters
  _TargetBlurring[0]      = size / 2.0;
  _TargetResolution[0][0] = size;
  _TargetResolution[0][1] = size;
  _TargetResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _TargetBlurring[i]      = _TargetBlurring[i-1] * 2;
    _TargetResolution[i][0] = _TargetResolution[i-1][0] * 2;
    _TargetResolution[i][1] = _TargetResolution[i-1][1] * 2;
    _TargetResolution[i][2] = _TargetResolution[i-1][2]; //* 2;
  }

  // Read source pixel size
  _source->GetPixelSize(&xsize, &ysize, &zsize);
  
  if (ysize<xsize)
    size = ysize;
  else
    size = xsize;
  //if (zsize<size)
    //size = zsize;

  // Default source parameters
  _SourceBlurring[0]      = size / 2.0;
  _SourceResolution[0][0] = size;
  _SourceResolution[0][1] = size;
  _SourceResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _SourceBlurring[i]      = _SourceBlurring[i-1] * 2;
    _SourceResolution[i][0] = _SourceResolution[i-1][0] * 2;
    _SourceResolution[i][1] = _SourceResolution[i-1][1] * 2;
    _SourceResolution[i][2] = _SourceResolution[i-1][2]; //* 2;
  }

  // Remaining parameters
  for (i = 0; i < _NumberOfLevels; i++) {
    _NumberOfIterations[i] = 20;
    _NumberOfSteps[i]      = 4;
    _LengthOfSteps[i]      = 2 * pow(2.0, i);
  }

  // Try to guess padding by looking at voxel values in all eight corners of the volume:
  // If all values are the same we assume that they correspond to the padding value
  _TargetPadding = MIN_GREY;
  if ((_target->Get(_target->GetX()-1, 0, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, 0, _target->GetZ()-1)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, 0)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, 0, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, _target->GetZ()-1) == _target->Get(0, 0, 0))) {
    _TargetPadding = _target->Get(0, 0, 0);
  }

  _SourcePadding = MIN_GREY;
  if ((_source->Get(_source->GetX()-1, 0, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, 0, _source->GetZ()-1)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, 0)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, 0, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, _source->GetZ()-1) == _source->Get(0, 0, 0))) {
    _SourcePadding = _source->Get(0, 0, 0);
  }
}

void irtkImageRigidRegistrationWithPadding_hybrid::GuessParameterThickSlicesNMI()
{
  int i;
  double xsize, ysize, zsize,size;

  if ((_target == NULL) || (_source == NULL)) {
    cerr << "irtkImageRigidRegistrationWithPadding_hybrid::GuessParameter: Target and source image not found" << endl;
    exit(1);
  }

  // Default parameters for registration
  _NumberOfLevels     = 3;
  _NumberOfBins       = 64;

  // Default parameters for optimization
  _SimilarityMeasure  = NMI;
  _OptimizationMethod = GradientDescent;
  _Epsilon            = 0.0001;

  // Read target pixel size
  _target->GetPixelSize(&xsize, &ysize, &zsize);
  
  if (ysize<xsize)
    size = ysize;
  else
    size = xsize;
  //if (zsize<size)
    //size = zsize;  

  // Default target parameters
  _TargetBlurring[0]      = size / 2.0;
  _TargetResolution[0][0] = size;
  _TargetResolution[0][1] = size;
  _TargetResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _TargetBlurring[i]      = _TargetBlurring[i-1] * 2;
    _TargetResolution[i][0] = _TargetResolution[i-1][0] * 2;
    _TargetResolution[i][1] = _TargetResolution[i-1][1] * 2;
    _TargetResolution[i][2] = _TargetResolution[i-1][2]; //* 2;
  }

  // Read source pixel size
  _source->GetPixelSize(&xsize, &ysize, &zsize);
  
  if (ysize<xsize)
    size = ysize;
  else
    size = xsize;
  //if (zsize<size)
    //size = zsize;

  // Default source parameters
  _SourceBlurring[0]      = size / 2.0;
  _SourceResolution[0][0] = size;
  _SourceResolution[0][1] = size;
  _SourceResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _SourceBlurring[i]      = _SourceBlurring[i-1] * 2;
    _SourceResolution[i][0] = _SourceResolution[i-1][0] * 2;
    _SourceResolution[i][1] = _SourceResolution[i-1][1] * 2;
    _SourceResolution[i][2] = _SourceResolution[i-1][2]; //* 2;
  }

  // Remaining parameters
  for (i = 0; i < _NumberOfLevels; i++) {
    _NumberOfIterations[i] = 20;
    _NumberOfSteps[i]      = 4;
    _LengthOfSteps[i]      = 2 * pow(2.0, i);
  }

  // Try to guess padding by looking at voxel values in all eight corners of the volume:
  // If all values are the same we assume that they correspond to the padding value
  _TargetPadding = MIN_GREY;
  if ((_target->Get(_target->GetX()-1, 0, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, 0, _target->GetZ()-1)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, 0)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, 0, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, _target->GetZ()-1) == _target->Get(0, 0, 0))) {
    _TargetPadding = _target->Get(0, 0, 0);
  }

  _SourcePadding = MIN_GREY;
  if ((_source->Get(_source->GetX()-1, 0, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, 0, _source->GetZ()-1)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, 0)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, 0, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, _source->GetZ()-1) == _source->Get(0, 0, 0))) {
    _SourcePadding = _source->Get(0, 0, 0);
  }
}

void irtkImageRigidRegistrationWithPadding_hybrid::GuessParameterSliceToVolume(bool useNMI)
{
  int i;
  double xsize, ysize, zsize;

  if ((_target == NULL) || (_source == NULL)) {
    cerr << "irtkImageRigidRegistrationWithPadding_hybrid::GuessParameter: Target and source image not found" << endl;
    exit(1);
  }

  // Default parameters for registration
  _NumberOfLevels     = 3;
  _NumberOfBins       = 64;

  // Default parameters for optimization
  _SimilarityMeasure  = CC; //NMI
  if(useNMI)
    _SimilarityMeasure  = NMI;
  _OptimizationMethod = GradientDescent;
  _Epsilon            = 0.0001;

  // Read target pixel size
  _target->GetPixelSize(&xsize, &ysize, &zsize);
  
  double size;
  
  if (ysize<xsize)
    size = ysize;
  else
    size = xsize;

  // Default target parameters
  _TargetBlurring[0]      = size / 2.0;
  _TargetResolution[0][0] = size;
  _TargetResolution[0][1] = size;
  _TargetResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _TargetBlurring[i]      = _TargetBlurring[i-1] * 2;
    _TargetResolution[i][0] = _TargetResolution[i-1][0] * 2;
    _TargetResolution[i][1] = _TargetResolution[i-1][1] * 2;
    _TargetResolution[i][2] = _TargetResolution[i-1][2];
  }

  // Read source pixel size
  _source->GetPixelSize(&xsize, &ysize, &zsize);
  
  if (ysize<xsize)
    size = ysize;
  else
    size = xsize;
  if (zsize<size)
    size = zsize;
  

  // Default source parameters
  _SourceBlurring[0]      = size / 2.0;
  _SourceResolution[0][0] = size;
  _SourceResolution[0][1] = size;
  _SourceResolution[0][2] = size;

  for (i = 1; i < _NumberOfLevels; i++) {
    _SourceBlurring[i]      = _SourceBlurring[i-1] * 2;
    _SourceResolution[i][0] = _SourceResolution[i-1][0] * 2;
    _SourceResolution[i][1] = _SourceResolution[i-1][1] * 2;
    _SourceResolution[i][2] = _SourceResolution[i-1][2] * 2;
  }

  // Remaining parameters
  for (i = 0; i < _NumberOfLevels; i++) {
    _NumberOfIterations[i] = 20;
    _NumberOfSteps[i]      = 4;
    _LengthOfSteps[i]      = 2 * pow(2.0, i);
  }

  // Try to guess padding by looking at voxel values in all eight corners of the volume:
  // If all values are the same we assume that they correspond to the padding value
  _TargetPadding = MIN_GREY;
  if ((_target->Get(_target->GetX()-1, 0, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, 0, _target->GetZ()-1)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, 0)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, 0, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, _target->GetZ()-1) == _target->Get(0, 0, 0))) {
    _TargetPadding = _target->Get(0, 0, 0);
  }

  _SourcePadding = MIN_GREY;
  if ((_source->Get(_source->GetX()-1, 0, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, 0, _source->GetZ()-1)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, 0)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, 0, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, _source->GetZ()-1) == _source->Get(0, 0, 0))) {
    _SourcePadding = _source->Get(0, 0, 0);
  }
}

void irtkImageRigidRegistrationWithPadding_hybrid::GuessParameterDistortion(double res)
{
  int i;
  double xsize, ysize, zsize, size;

  if ((_target == NULL) || (_source == NULL)) {
    cerr << "irtkImageRigidRegistrationWithPadding_hybrid::GuessParameter: Target and source image not found" << endl;
    exit(1);
  }

  // Default parameters for registration
  _NumberOfLevels     = 1;
  _NumberOfBins       = 64;

  // Default parameters for optimization
  _SimilarityMeasure  = CC;
  _OptimizationMethod = GradientDescent;
  _Epsilon            = 0.0001;

  // Read target pixel size
  _target->GetPixelSize(&xsize, &ysize, &zsize);
  
  if(res<=0)
  {
    if (ysize<xsize)
      size = ysize;
    else
      size = xsize;
  }
  else size = res;

  // Default target parameters
  _TargetBlurring[0]      = 0;//size / 2.0;
  _TargetResolution[0][0] = size;
  _TargetResolution[0][1] = size;
  _TargetResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _TargetBlurring[i]      = _TargetBlurring[i-1] * 2;
    _TargetResolution[i][0] = _TargetResolution[i-1][0] * 2;
    _TargetResolution[i][1] = _TargetResolution[i-1][1] * 2;
    _TargetResolution[i][2] = _TargetResolution[i-1][2];
  }

  // Read source pixel size
  _source->GetPixelSize(&xsize, &ysize, &zsize);
  
  if(res<=0)
  {
    if (ysize<xsize)
      size = ysize;
    else
      size = xsize;
  }
  else size = res;

  // Default source parameters
  _SourceBlurring[0]      = 0; //size / 2.0;
  _SourceResolution[0][0] = size;
  _SourceResolution[0][1] = size;
  _SourceResolution[0][2] = zsize;

  for (i = 1; i < _NumberOfLevels; i++) {
    _SourceBlurring[i]      = _SourceBlurring[i-1] * 2;
    _SourceResolution[i][0] = _SourceResolution[i-1][0] * 2;
    _SourceResolution[i][1] = _SourceResolution[i-1][1] * 2;
    _SourceResolution[i][2] = _SourceResolution[i-1][2];
  }

  // Remaining parameters
  for (i = 0; i < _NumberOfLevels; i++) {
    _NumberOfIterations[i] = 20;
    _NumberOfSteps[i]      = 4;
    _LengthOfSteps[i]      = 2 * pow(2.0, i);
  }

  // Try to guess padding by looking at voxel values in all eight corners of the volume:
  // If all values are the same we assume that they correspond to the padding value
  _TargetPadding = MIN_GREY;
  if ((_target->Get(_target->GetX()-1, 0, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, 0)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, 0, _target->GetZ()-1)                                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, 0)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(0, _target->GetY()-1, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, 0, _target->GetZ()-1)                 == _target->Get(0, 0, 0)) &&
      (_target->Get(_target->GetX()-1, _target->GetY()-1, _target->GetZ()-1) == _target->Get(0, 0, 0))) {
    _TargetPadding = _target->Get(0, 0, 0);
  }

  _SourcePadding = MIN_GREY;
  if ((_source->Get(_source->GetX()-1, 0, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, 0)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, 0, _source->GetZ()-1)                                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, 0)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(0, _source->GetY()-1, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, 0, _source->GetZ()-1)                 == _source->Get(0, 0, 0)) &&
      (_source->Get(_source->GetX()-1, _source->GetY()-1, _source->GetZ()-1) == _source->Get(0, 0, 0))) {
    _SourcePadding = _source->Get(0, 0, 0);
  }
  cerr<<endl<<endl;
  cerr<<"GuessParameterDistortion:"<<endl;
  cerr<<"Target padding is "<<_TargetPadding<<endl;
  cerr<<"Source padding is "<<_SourcePadding<<endl;
  cerr<<endl<<endl;
}

void irtkImageRigidRegistrationWithPadding_hybrid::freeGPU()
{
  ccCostFunction_gpu->freeGPU();
}

void irtkImageRigidRegistrationWithPadding_hybrid::SetSource(irtkGreyImage *source)
{
  //TODO: they come in with different sizes after initialization!
  _source = source;
  //ccCostFunction_gpu->source_size = make_uint3(_source->GetX(), _source->GetY(), _source->GetZ());
  ccCostFunction_gpu->setSource(_source->GetPointerToVoxels(), 
    make_uint3(_source->GetX(), _source->GetY(), _source->GetZ()),
    make_float3(_source->GetXSize(), _source->GetYSize(), _source->GetZSize()));
  //TODO cast on GPU
 /* irtkGenericImage<float> im_source = *_source;

  if (d_source != NULL)
    checkCudaErrors(cudaFree(d_source));
  checkCudaErrors(cudaMalloc(&d_source, sizeof(float) * im_source.GetNumberOfVoxels()));
  //not every time
  checkCudaErrors(cudaMemcpy(d_source, im_source.GetPointerToVoxels(), im_source.GetNumberOfVoxels()*sizeof(float), cudaMemcpyHostToDevice));
  */
  //ccCostFunction_gpu->prepareReconTex();
}

void irtkImageRigidRegistrationWithPadding_hybrid::SetTarget(irtkGreyImage *target)
{
  _target = target;
  //ccCostFunction_gpu->source_size = make_uint3(_target->GetX(), _target->GetY(), _target->GetZ());
  ccCostFunction_gpu->setTarget(_target->GetPointerToVoxels(), make_uint3(_target->GetX(), _target->GetY(), _target->GetZ()));

  //TODO cast on GPU
/*  irtkGenericImage<float> im_target = *_target;

  if (d_target != NULL)
    checkCudaErrors(cudaFree(d_target));
  checkCudaErrors(cudaMalloc(&d_target, sizeof(float) * im_target.GetNumberOfVoxels()));
  //every time
  checkCudaErrors(cudaMemcpy(d_target, im_target.GetPointerToVoxels(), im_target.GetNumberOfVoxels()*sizeof(float), cudaMemcpyHostToDevice));
  */
}


void irtkImageRigidRegistrationWithPadding_hybrid::UpdateGPU()
{
  //printf("UpdateGPU() \n");
  transformations transf;
  transf.targetI2W = toMatrix4<float>(_target->GetImageToWorldMatrix());
  transf.targetW2I = toMatrix4<float>(_target->GetWorldToImageMatrix());
  transf.sourceI2W = toMatrix4<float>(_source->GetImageToWorldMatrix());
  transf.sourceW2I = toMatrix4<float>(_source->GetWorldToImageMatrix());
  irtkRigidTransformation* trans_ = dynamic_cast<irtkRigidTransformation*>(_transformation);
  trans_->UpdateMatrix();
  transf.Tmat = toMatrix4<float>(trans_->GetMatrix());
  transf.tsize.x = _target->GetX();
  transf.tsize.y = _target->GetY();
  transf.tsize.z = _target->GetZ();
  transf.ssize.x = _source->GetX();
  transf.ssize.y = _source->GetY();
  transf.ssize.z = _source->GetZ();

  ccCostFunction_gpu->setTransformations(transf);

  //last resort: cpy target source here
  ccCostFunction_gpu->setSource(_source->GetPointerToVoxels(), 
    make_uint3(_source->GetX(), _source->GetY(), _source->GetZ()),
    make_float3(_source->GetXSize(), _source->GetYSize(), _source->GetZSize()));
  ccCostFunction_gpu->setTarget(_target->GetPointerToVoxels(), 
    make_uint3(_target->GetX(), _target->GetY(), _target->GetZ()));

}


double irtkImageRigidRegistrationWithPadding_hybrid::Evaluate()
{
  double result = 0.0;
#if 0

  irtkRigidTransformation* trans_ = dynamic_cast<irtkRigidTransformation*>(_transformation);
  trans_->UpdateMatrix();
  ccCostFunction_gpu->m_t.Tmat = toMatrix4<float>(trans_->GetMatrix());

  result = ccCostFunction_gpu->evaluate();

#else

//  #ifndef HAS_TBB
  int i, j, k, t;
//  #endif

  // Pointer to reference data
  irtkGreyPixel *ptr2target;

  // Print debugging information
  this->Debug("irtkImageRigidRegistrationWithPadding_hybrid::Evaluate");

  // Invert transformation
  //((irtkRigidTransformation *)_transformation)->Invert();

  // Create iterator
  irtkHomogeneousTransformationIterator
  iterator((irtkHomogeneousTransformation *)_transformation);

  // Initialize metric
  _metric->Reset();

  // Pointer to voxels in target image
  ptr2target = _target->GetPointerToVoxels();
  
// #ifdef HAS_TBB
//   irtkMultiThreadedImageRigidRegistrationWithPaddingEvaluate evaluate(this);
//   parallel_reduce(blocked_range<int>(0, _target->GetZ(), 20), evaluate);
// #else


  for (t = 0; t < _target->GetT(); t++) {

    // Initialize iterator
    iterator.Initialize(_target, _source);

    // Loop over all voxels in the target (reference) volume
    for (k = 0; k < _target->GetZ(); k++) {
      for (j = 0; j < _target->GetY(); j++) {
        for (i = 0; i < _target->GetX(); i++) {
          // Check whether reference point is valid
          if (*ptr2target >= 0) {
            // Check whether transformed point is inside source volume
            if ((iterator._x > _source_x1) && (iterator._x < _source_x2) &&
                (iterator._y > _source_y1) && (iterator._y < _source_y2) &&
                (iterator._z > _source_z1) && (iterator._z < _source_z2)) {
              // Add sample to metric. Note: only linear interpolation supported at present
	      //double value = (static_cast<irtkLinearInterpolateImageFunction*> (_interpolator))->EvaluateWithPadding(-1,iterator._x, iterator._y, iterator._z, t);
	      double value = _interpolator->EvaluateInside(iterator._x, iterator._y, iterator._z, t);
	      if (value >= 0)
                _metric->Add(*ptr2target, round(value));
            }
            iterator.NextX();
          } else {
            // Advance iterator by offset
            iterator.NextX(*ptr2target * -1);
            i          -= (*ptr2target) + 1;
            ptr2target -= (*ptr2target) + 1;
          }
          ptr2target++;
        }
        iterator.NextY();
      }
      iterator.NextZ();
    }
  }

//  #endif


  // Invert transformation
  //((irtkRigidTransformation *)_transformation)->Invert();

  // Evaluate similarity measure
  result = _metric->Evaluate();
  //printf(" %f ", result);

  
#endif

  //TODO debug until same
  //printf(" %f %f \n", test_result,  result);

  //return result;
  return result;// ;
}

void irtkImageRigidRegistrationWithPadding_hybrid::Run()
{
  int i, j, level;
  char buffer[256];
  double step, epsilon = 0, delta, maxChange = 0;

  // Print debugging information
  this->Debug("irtkImageRegistration::Run");

  if (_source == NULL) {
    cerr << "Registration::Run: Filter has no source input" << endl;
    exit(1);
  }

  if (_target == NULL) {
    cerr << "Registration::Run: Filter has no target input" << endl;
    exit(1);
  }

  if (_transformation == NULL) {
    cerr << "irtkImageRegistration::Run: Filter has no transformation output" << endl;
    exit(1);
  }

  // Do the initial set up for all levels
  irtkImageRegistration::Initialize();

  // Loop over levels
  for (level = _NumberOfLevels - 1; level >= 0; level--) {


    // Initial step size
    step = _LengthOfSteps[level];

    // Print resolution level
    cout << "Resolution level no. " << level + 1 << " (step sizes ";
    cout << step << " to " << step / pow(2.0, static_cast<double>(_NumberOfSteps[level] - 1)) << ")\n";

    // Initial Delta
    delta = _Delta[level];
    cout << "Delta values : " << delta << " to ";
    cout << delta / pow(2.0, static_cast<double>(_NumberOfSteps[level] - 1)) << "\n";

#ifdef HISTORY
    history->Clear();
#endif

    // Initialize for this level
    this->Initialize(level);
    this->UpdateGPU();

    // Save pre-processed images if we are debugging
    if (_DebugFlag == true){
      sprintf(buffer, "source_%d.nii.gz", level);
      _source->Write(buffer);
    }
    if (_DebugFlag == true){
      sprintf(buffer, "target_%d.nii.gz", level);
      _target->Write(buffer);
    }

#ifdef HAS_TBB
    task_scheduler_init init(tbb_no_threads);
#if USE_TIMING
    tick_count t_start = tick_count::now();
#endif
#endif

    // Run the registration filter at this resolution
    for (i = 0; i < _NumberOfSteps[level]; i++) {
      for (j = 0; j < _NumberOfIterations[level]; j++) {
        cout << "Iteration = " << j + 1 << " (out of " << _NumberOfIterations[level];
        cout << "), step size = " << step << endl;

        // Optimize at lowest level of resolution
        _optimizer->SetStepSize(step);
        _optimizer->SetEpsilon(_Epsilon);
        _optimizer->Run(epsilon, maxChange);

        // Check whether we made any improvement or not
        if (epsilon > _Epsilon && maxChange > delta) {
          sprintf(buffer, "log_%.3d_%.3d_%.3d.dof", level, i + 1, j + 1);
          if (_DebugFlag == true) _transformation->Write(buffer);
          this->Print();
        }
        else {
          sprintf(buffer, "log_%.3d_%.3d_%.3d.dof", level, i + 1, j + 1);
          if (_DebugFlag == true) _transformation->Write(buffer);
          this->Print();
          break;
        }
      }
      step = step / 2;
      delta = delta / 2.0;
    }

#ifdef HAS_TBB
#if USE_TIMING
    tick_count t_end = tick_count::now();
    if (tbb_debug) cout << this->NameOfClass() << " = " << (t_end - t_start).seconds() << " secs." << endl;
#endif
    init.terminate();

#endif

    // Do the final cleaning up for this level
    this->Finalize(level);

#ifdef HISTORY
    history->Print();
#endif

  }

  // Do the final cleaning up for all levels
  this->Finalize();
}
