/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkPointFreeFormRegistration.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKPOINTFREEFORMREGISTRATION_H

#define _IRTKPOINTFREEFORMREGISTRATION_H

#include <irtkImage.h>
#include <irtkPointRegistration.h>
#include <irtkTransformation.h>

class irtkPointFreeFormRegistration : public irtkPointRegistration
{

protected:

  /// Pointer to the local transformation which is currently optimized
  irtkBSplineFreeFormTransformation *_affd;

  /// Pointer to the global transformation which is constant
  irtkMultiLevelFreeFormTransformation *_mffd;

  /// Initial set up for the registration
  virtual void Initialize();

  /// Final set up for the registration
  virtual void Finalize();

  /// Optimize registration
  virtual void Optimize();

public:

  /// Constructor
  irtkPointFreeFormRegistration();

  /// Destructor
  virtual ~irtkPointFreeFormRegistration();

  /// Sets output for the registration filter
  virtual void SetOutput(irtkTransformation *);

  /// Run the filter
  virtual void Run();

  /// Returns the name of the class
  virtual const char *NameOfClass();

};

#endif
