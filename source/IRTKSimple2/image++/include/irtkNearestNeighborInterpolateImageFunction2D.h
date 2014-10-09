/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkNearestNeighborInterpolateImageFunction2D.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKNEARESTNEIGHBORINTERPOLATEIMAGEFUNCTION2D_H

#define _IRTKNEARESTNEIGHBORINTERPOLATEIMAGEFUNCTION2D_H

/**
 * Class for nearest neighbor interpolation of images
 *
 * This class defines and implements the nearest neighbor interpolation of
 * images.
 */

class irtkNearestNeighborInterpolateImageFunction2D : public irtkInterpolateImageFunction
{

private:

  /// Dimension of input image in X-direction
  int _x;

  /// Dimension of input image in Y-direction
  int _y;

public:

  /// Constructor
  irtkNearestNeighborInterpolateImageFunction2D();

  /// Destructor
  ~irtkNearestNeighborInterpolateImageFunction2D();

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Initialize
  virtual void Initialize();

/// Evaluate
  virtual double Evaluate(double, double, double = 0, double = 0);

  /** Evaluate the filter at an arbitrary image location (in pixels) without
   *  handling boundary conditions. This version is faster than the method
   *  above, but is only defined inside the image domain. */
  virtual double EvaluateInside(double, double, double = 0, double = 0);

};

#endif
