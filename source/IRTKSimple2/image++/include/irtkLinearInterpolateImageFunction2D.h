/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkLinearInterpolateImageFunction2D.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKLINEARINTERPOLATEIMAGEFUNCTION2D_H

#define _IRTKLINEARINTERPOLATEIMAGEFUNCTION2D_H

/**
 * Class for linear interpolation of images
 *
 * This class defines and implements the linear interpolation of
 * images.
 */

class irtkLinearInterpolateImageFunction2D : public irtkInterpolateImageFunction
{

private:

  /// Dimension of input image in X-direction
  int _x;

  /// Dimension of input image in Y-direction
  int _y;

  /// Offsets for fast pixel access
  int _offset1, _offset2, _offset3, _offset4;

public:

  /// Constructor
  irtkLinearInterpolateImageFunction2D();

  /// Destructor
  ~irtkLinearInterpolateImageFunction2D();

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Initialize
  virtual void Initialize();

  /// Evaluate the filter at an arbitrary image location (in pixels)
  virtual double Evaluate(double, double, double = 0, double = 0);

  /** Evaluate the filter at an arbitrary image location (in pixels) without
   *  handling boundary conditions. This version is faster than the method
   *  above, but is only defined inside the image domain. */
  virtual double EvaluateInside(double, double, double = 0, double = 0);

};

#endif
