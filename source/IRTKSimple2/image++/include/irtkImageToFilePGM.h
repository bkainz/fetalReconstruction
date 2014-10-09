/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageToFilePGM.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKIMAGETOFILEPGM_H

#define _IRTKIMAGETOFILEPGM_H

/**
 * Class for image to PGM file filter.
 *
 * This is a class which takes an image as input and produces an image file
 * in PGM file format. Note that PGM file formats support only 2D images!!!
 */

class irtkImageToFilePGM : public irtkImageToFile
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
