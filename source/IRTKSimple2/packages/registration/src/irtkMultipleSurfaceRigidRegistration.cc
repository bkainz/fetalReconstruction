/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkMultipleSurfaceRigidRegistration.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifdef HAS_VTK

#include <irtkMultipleSurfaceRegistration.h>

irtkMultipleSurfaceRigidRegistration::irtkMultipleSurfaceRigidRegistration (): irtkMultipleSurfaceRegistration()
{
}

const char *irtkMultipleSurfaceRigidRegistration::NameOfClass ()
{
  return "irtkMultipleSurfaceRigidRegistration";
}

void irtkMultipleSurfaceRigidRegistration::SetOutput (irtkTransformation * transformation)
{
  if (strcmp (transformation->NameOfClass (), "irtkRigidTransformation") != 0) {
    cerr << this->NameOfClass ()
         << "::SetOutput: Transformation must be rigid" << endl;
    exit (0);
  }
  _transformation = transformation;
}

void irtkMultipleSurfaceRigidRegistration::Initialize ()
{
  // Initialize base class
  this->irtkMultipleSurfaceRegistration::Initialize ();

  // Create point-based registration
  _preg = new irtkPointRigidRegistration;
}

void irtkMultipleSurfaceRigidRegistration::Finalize ()
{
  // Delete point registration
  delete _preg;
}

#endif
