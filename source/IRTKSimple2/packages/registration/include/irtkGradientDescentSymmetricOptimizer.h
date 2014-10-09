/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkGradientDescentSymmetricOptimizer.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKGRADIENTDESCENTSYMMETRICOPTIMIZER_H

#define _IRTKGRADIENTDESCENTSYMMETRICOPTIMIZER_H

/**
 * Generic class for gradient descent optimization of voxel-based
 * symmetric registration.
 */

class irtkGradientDescentSymmetricOptimizer : public irtkSymmetricOptimizer
{

public:

  /// Run the optimizer
  virtual double Run();

  /// Print name of the class
  virtual const char *NameOfClass();

};

inline const char *irtkGradientDescentSymmetricOptimizer::NameOfClass()
{
  return "irtkGradientDescentSymmetricOptimizer";
}

#endif
