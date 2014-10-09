/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkMultipleSurfaceAffineRegistration.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifdef HAS_VTK

#include <irtkMultipleSurfaceRegistration.h>

irtkMultipleSurfaceAffineRegistration::irtkMultipleSurfaceAffineRegistration (): irtkMultipleSurfaceRegistration()
{
}

const char *irtkMultipleSurfaceAffineRegistration::NameOfClass ()
{
  return "irtkMultipleSurfaceAffineRegistration";
}

void irtkMultipleSurfaceAffineRegistration::SetOutput (irtkTransformation * transformation)
{
  if (strcmp (transformation->NameOfClass (), "irtkAffineTransformation") != 0) {
    cerr << this->NameOfClass ()
         << "::SetOutput: Transformation must be affine" << endl;
    exit (0);
  }
  _transformation = transformation;
}

void irtkMultipleSurfaceAffineRegistration::Initialize ()
{
  // Initialize base class
  this->irtkMultipleSurfaceRegistration::Initialize ();

  // Create point-based registration
  _preg = new irtkPointAffineRegistration;
}

void irtkMultipleSurfaceAffineRegistration::Finalize()
{
  // Delete point registration
  delete _preg;
}

#endif
