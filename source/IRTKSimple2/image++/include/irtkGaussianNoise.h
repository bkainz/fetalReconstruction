/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkGaussianNoise.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKGAUSSIANNOISE_H

#define _IRTKGAUSSIANNOISE_H

/**
 * Class for adding Gaussian noise to images
 *
 * This class implements a filter for adding Gaussian noise to images.
 *
 */

template <class VoxelType> class irtkGaussianNoise : public irtkNoise<VoxelType>
{

protected:

  /// The mean value of the noise distribution.
  double _Mean;

  /** The standard deviation of the noise distribution. This is the same as
      the amplitude of the noise.
  */
  double _Sigma;

  /// Minimum voxel value
  VoxelType _MinVal;

  /// Maximum voxel value
  VoxelType _MaxVal;

  // Returns the name of the class
  virtual const char *NameOfClass();

public:

  // Default constructor
  irtkGaussianNoise();

  /** Constructor.  Sets mean value and standard deviation of the noise
   *  distribution, and range of values permitted for each voxel in the image.
   */
  irtkGaussianNoise(double mean, double sigma, VoxelType min, VoxelType max);

  /// Destructor (empty).
  ~irtkGaussianNoise() {};

  // Run gaussian noise filter
  virtual double Run(int, int, int, int);

  /// Set mean of Gaussian noise
  SetMacro(Mean, double);

  /// Get mean of Gaussian noise
  GetMacro(Mean, double);

  /// Set standard deviation of Gaussian noise
  SetMacro(Sigma, double);

  /// Get standard deviation of Gaussian noise
  GetMacro(Sigma, double);

  /// Set minimum value for Gaussian noise
  SetMacro(MinVal, VoxelType);

  /// Get minimum value for Gaussian noise
  GetMacro(MinVal, VoxelType);

  /// Set maximum value for Gaussian noise
  SetMacro(MaxVal, VoxelType);

  /// Get maximum value for Gaussian noise
  GetMacro(MaxVal, VoxelType);
};

#endif
