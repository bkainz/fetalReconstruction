/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkSymmetricOptimizer.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKSYMMETRICOPTIMIZER_H

#define _IRTKSYMMETRICOPTIMIZER_H

/// Forward declaration
class irtkRegistration;

/**
 * Generic class for optimization of voxel-based registration.
 *
 * This class implements an optimizer which maximises voxel similarity
 * measures as used in image registration. This is the abstract base class
 * which defines a common interface for arbitrary optimization filters. Each
 * derived class has to implement all abstract member functions.
 *
 */

class irtkSymmetricOptimizer : public irtkObject
{

protected:

  /// Pointer to transformation
  irtkTransformation *_Transformation1;

  /// Pointer to transformation
  irtkTransformation *_Transformation2;

  /// Pointer to registration
  irtkRegistration *_Registration;

  /// Step size
  double _StepSize;

  /// Epsilon
  double _Epsilon;

  /// Storage for monitoring change in the transformation.
  double *transformationParams;

public:

  /// Constructor
  irtkSymmetricOptimizer();

  /// Destructor
  virtual ~irtkSymmetricOptimizer();

  /// Run the optimizer
  virtual double Run() = 0;

  /// Run the optimizer
  virtual void Run(double &, double &);

  /// Print name of the class
  virtual const char *NameOfClass() = 0;

  virtual void SetTransformation(irtkTransformation *, irtkTransformation *);

  virtual GetMacro(Transformation1, irtkTransformation *);

  virtual GetMacro(Transformation2, irtkTransformation *);

  virtual SetMacro(Registration, irtkRegistration *)

  virtual GetMacro(Registration, irtkRegistration *);

  virtual SetMacro(StepSize, double);

  virtual GetMacro(StepSize, double);

  virtual SetMacro(Epsilon, double);

  virtual GetMacro(Epsilon, double);

};

inline void irtkSymmetricOptimizer::SetTransformation(irtkTransformation *transformation1, irtkTransformation *transformation2)
{
  int n;

  _Transformation1 = transformation1;
  n  = _Transformation1->NumberOfDOFs();

  _Transformation2 = transformation2;
  n += _Transformation2->NumberOfDOFs();

  transformationParams = new double[n];
}

#include <irtkGradientDescentSymmetricOptimizer.h>

#endif
