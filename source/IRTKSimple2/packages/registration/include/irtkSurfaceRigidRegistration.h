/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkSurfaceRigidRegistration.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifdef HAS_VTK

#ifndef _IRTKSURFACERIGIDREGISTRATION_H

#define _IRTKSURFACERIGIDREGISTRATION_H

#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkPointRegistration.h>
#include <irtkSurfaceRegistration.h>

#include <vtkCellLocator.h>
#include <vtkPointLocator.h>
#include <irtkLocator.h>
#include <vtkFeatureEdges.h>

class irtkSurfaceRigidRegistration : public irtkSurfaceRegistration
{

protected:

  /// Initial set up for the registration
  virtual void Initialize();

  /// Final set up for the registration
  virtual void Finalize();

public:

  /// Constructor
  irtkSurfaceRigidRegistration();

  /// Sets output for the registration filter
  virtual void SetOutput(irtkTransformation *);

  /// Returns the name of the class
  virtual const char *NameOfClass();

};

#endif

#endif
