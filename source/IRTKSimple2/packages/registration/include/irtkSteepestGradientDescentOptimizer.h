/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkSteepestGradientDescentOptimizer.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKSTEEPESTGRADIENTDESCENTOPTIMIZER_H

#define _IRTKSTEEPESTGRADIENTDESCENTOPTIMIZER_H

/**
 * Generic class for steepest gradient descent optimization of voxel-based
 * registration.
 */

class irtkSteepestGradientDescentOptimizer : public irtkOptimizer
{

public:

  /// Run the optimizer
  virtual double Run();

  /// Print name of the class
  virtual const char *NameOfClass();

};

inline const char *irtkSteepestGradientDescentOptimizer::NameOfClass()
{
  return "irtkSteepestGradientDescentOptimizer";
}

#endif
