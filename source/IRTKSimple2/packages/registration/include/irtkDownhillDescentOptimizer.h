/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkDownhillDescentOptimizer.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKDOWNHILLDESCENTOPTIMIZER_H

#define _IRTKDOWNHILLDESCENTOPTIMIZER_H

/**
 * Generic class for downhill descent optimization of voxel-based
 * registration.
 */

class irtkDownhillDescentOptimizer : public irtkOptimizer
{

public:

  /// Optimization method
  virtual double Run();

  /// Print name of the class
  virtual const char *NameOfClass();
};

inline const char *irtkDownhillDescentOptimizer::NameOfClass()
{
  return "irtkDownhillGradientDescentOptimizer";
}

#endif
