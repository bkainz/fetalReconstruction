/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkUniformNoise.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKUNIFORMNOISE_H

#define _IRTKUNIFORMNOISE_H

/**
 * Class for adding uniform noise to images
 *
 * This class implements a filter for adding uniformly distributed noise to
 * images.
 *
 */

template <class VoxelType> class irtkUniformNoise : public irtkNoise<VoxelType>
{

protected:

  /// Returns the name of the class
  virtual const char *NameOfClass();

public:

  /// Constructor
  irtkUniformNoise(double amplitude = 1);

  /// Destructor (empty).
  ~irtkUniformNoise() {};

  /// Run uniform noise filter
  virtual double Run(int, int, int, int);

};

#endif
