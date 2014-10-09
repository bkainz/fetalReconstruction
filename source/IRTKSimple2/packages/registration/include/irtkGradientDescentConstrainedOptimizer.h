/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkGradientDescentConstrainedOptimizer.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKGRADIENTDESCENTCONSTRAINEDOPTIMIZER_H

#define _IRTKGRADIENTDESCENTCONSTRAINEDOPTIMIZER_H

/**
 * Class for gradient descent optimization of voxel-based
 * registration which enforces hard limits on the registration
 * parameters
 */

class irtkGradientDescentConstrainedOptimizer : public irtkOptimizer
{

  /// Hard limits
  double _limits;

public:

  /// Constructor
  irtkGradientDescentConstrainedOptimizer();

  /// Run the optimizer
  virtual double Run();

  /// Print name of the class
  virtual const char *NameOfClass();

  /// Set hard limits for parameters
  virtual void SetLimits(double);

};

inline const char *irtkGradientDescentConstrainedOptimizer::NameOfClass()
{
  return "irtkGradientDescentOptimizer";
}

inline void irtkGradientDescentConstrainedOptimizer::SetLimits(double limits)
{
  _limits = limits;
}

#endif
