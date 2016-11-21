/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageRigidRegistration.h 509 2012-01-17 10:45:53Z mm3 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2012-01-17 10:45:53 +0000 (Tue, 17 Jan 2012) $
  Version   : $Revision: 509 $
  Changes   : $Author: mm3 $

=========================================================================*/

#ifndef _irtkImageRigidRegistrationWithPadding_hybrid_HYBRID_H

#define _irtkImageRigidRegistrationWithPadding_hybrid_HYBRID_H

/**
 * Filter for rigid registration based on voxel similarity measures extended by source padding.
 */
#include <irtkImageRegistrationWithPadding.h>

#include "ccSimilarityMetric.cuh"

class irtkImageRigidRegistrationWithPadding_hybrid : public irtkImageRegistrationWithPadding
{

protected:

  /// Evaluate the similarity measure for a given transformation.
  virtual double Evaluate();

  //// Initial set up for the registration
  //virtual void Initialize();

  //// Final set up for the registration
  //virtual void Finalize();

  ccSimilarityMetric* ccCostFunction_gpu;

public:

  irtkImageRigidRegistrationWithPadding_hybrid();
  ~irtkImageRigidRegistrationWithPadding_hybrid();

  //needed only once
  virtual void SetSource(irtkGreyImage *source);
  virtual void SetTarget(irtkGreyImage *target);
  void freeGPU();

  /// Runs the registration filter
  virtual void Run();
  virtual void UpdateGPU();

  //strange: result changes slightly if overwritten in library
  /*virtual void SetInput(irtkGreyImage *target, irtkGreyImage *source);

  virtual void Initialize(int level);*/

  /** Sets the output for the registration filter. The output must be a rigid
   *  transformation. The current parameters of the rigid transformation are
   *  used as initial guess for the rigid registration. After execution of the
   *  filter the parameters of the rigid transformation are updated with the
   *  optimal transformation parameters.
   */
  virtual void SetOutput(irtkTransformation *);

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Print information about the progress of the registration
  virtual void Print();

  /// Guess parameters
  virtual void GuessParameter();
  /// Guess parameters for slice to volume registration 
  virtual void GuessParameterSliceToVolume(bool useNMI = false);
  /// Guess parameters volumes with thick slices
  virtual void GuessParameterThickSlices();
    /// Guess parameters volumes with thick slices and NMI
  virtual void GuessParameterThickSlicesNMI();
  /// Guess parameters for distortion correction
  virtual void GuessParameterDistortion(double res);

};

inline void irtkImageRigidRegistrationWithPadding_hybrid::SetOutput(irtkTransformation *transformation)
{
  if (strcmp(transformation->NameOfClass(), "irtkRigidTransformation") != 0) {
    cerr << "irtkImageRigidRegistration::SetOutput: Transformation must be rigid"
         << endl;
    exit(0);
  }
  _transformation = transformation;
}

inline const char *irtkImageRigidRegistrationWithPadding_hybrid::NameOfClass()
{
  return "irtkImageRigidRegistrationWithPadding_hybrid";
}

inline void irtkImageRigidRegistrationWithPadding_hybrid::Print()
{
  _transformation->Print();
}

#endif
