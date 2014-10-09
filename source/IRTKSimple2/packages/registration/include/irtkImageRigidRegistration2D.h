/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageRigidRegistration2D.h 304 2011-04-10 21:46:30Z ws207 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2011-04-10 22:46:30 +0100 (Sun, 10 Apr 2011) $
  Version   : $Revision: 304 $
  Changes   : $Author: ws207 $

=========================================================================*/

#ifndef _IRTKIMAGERIGIDREGISTRATION2D_H

#define _IRTKIMAGERIGIDREGISTRATION2D_H

/**
 * Filter for rigid registration based on voxel similarity measures.
 *
 * This class implements a registration filter for the rigid registration of
 * two images. The basic algorithm is described in Studholme, Medical Image
 * Analysis, Vol. 1, No. 2, 1996.
 *
 */

class irtkImageRigidRegistration2D : public irtkImageRigidRegistration
{

protected:

  /// Evaluate the similarity measure for a given transformation.
  virtual double Evaluate();

public:

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Guess parameters
  virtual void GuessParameter();

};

inline const char *irtkImageRigidRegistration2D::NameOfClass()
{
  return "irtkImageRigidRegistration2D";
}

#endif
