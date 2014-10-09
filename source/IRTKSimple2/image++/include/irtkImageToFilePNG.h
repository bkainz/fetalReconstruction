/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageToFilePNG.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKIMAGETOFILEPNG_H

#define _IRTKIMAGETOFILEPNG_H

#ifdef HAS_PNG

/**
 * Class for image to PNG file filter.
 *
 * This is a class which takes an image as input and produces an image file
 * in PNG file format. Note that PNG file formats support only 2D images!!!
 */

class irtkImageToFilePNG : public irtkImageToFile
{

protected:

  /// Initialize filter
  virtual void Initialize();

public:

  /// Write entire image
  virtual void Run();

  /// Return name of class
  virtual const char *NameOfClass();

};

#endif

#endif
