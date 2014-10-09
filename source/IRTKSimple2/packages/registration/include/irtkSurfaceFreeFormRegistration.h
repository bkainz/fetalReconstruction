/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkSurfaceFreeFormRegistration.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifdef HAS_VTK

#ifndef _IRTKSURFACEFREEFORMREGISTRATION_H

#define _IRTKSURFACEFREEFORMREGISTRATION_H

class irtkSurfaceFreeFormRegistration : public irtkSurfaceRegistration
{

protected:

  /// Initial set up for the registration
  virtual void Initialize();

  /// Final set up for the registration
  virtual void Finalize();

public:

  /// Constructor
  irtkSurfaceFreeFormRegistration();

  /// Sets output for the registration filter
  virtual void SetOutput(irtkTransformation *);

  /// Returns the name of the class
  virtual const char *NameOfClass();
};

#endif

#endif
