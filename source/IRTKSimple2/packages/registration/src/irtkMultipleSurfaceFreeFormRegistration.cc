/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkMultipleSurfaceFreeFormRegistration.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifdef HAS_VTK

#include <irtkMultipleSurfaceRegistration.h>

irtkMultipleSurfaceFreeFormRegistration::irtkMultipleSurfaceFreeFormRegistration (): irtkMultipleSurfaceRegistration()
{
}

const char *irtkMultipleSurfaceFreeFormRegistration::NameOfClass ()
{
  return "irtkMultipleSurfaceFreeFormRegistration";
}

void irtkMultipleSurfaceFreeFormRegistration::SetOutput(irtkTransformation *transformation)
{
  if (strcmp(transformation->NameOfClass(),
             "irtkMultiLevelFreeFormTransformation") != 0) {
    cerr << "irtkMultipleSurfaceFreeFormRegistration::SetOutput: Transformation must be "
         << "irtkMultiLevelFreeFormTransformation" << endl;
    exit(0);
  }
  _transformation = transformation;
}

void irtkMultipleSurfaceFreeFormRegistration::Initialize ()
{
  // Initialize base class
  this->irtkMultipleSurfaceRegistration::Initialize ();

  // Create point-based registration
  _preg = new irtkPointFreeFormRegistration;
}

void irtkMultipleSurfaceFreeFormRegistration::Finalize ()
{
  // Delete point registration
  delete _preg;
}

#endif
