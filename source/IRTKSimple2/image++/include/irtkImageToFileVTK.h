/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageToFileVTK.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKIMAGETOFILEVTK_H

#define _IRTKIMAGETOFILEVTK_H

/**
 * Class for image to VTK file filter.
 *
 * This is a class which takes an image as input and produces an image file
 * in VTK file format.
 */

class irtkImageToFileVTK : public irtkImageToFile
{

protected:

  /// Initialize filter
  virtual void Initialize();

public:

  /// Return name of class
  virtual const char *NameOfClass();

};

#endif
