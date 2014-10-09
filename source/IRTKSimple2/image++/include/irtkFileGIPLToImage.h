/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkFileGIPLToImage.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKFILEGIPLTOIMAGE_H

#define _IRTKFILEGIPLTOIMAGE_H

/**
 * Class for reading images in GIPL file format.
 *
 * This is a class which reads images in GIPL file format and converts them
 * into images. The GIPL file format is a file format for 3D images. It has
 * been used in Guy's Hospital as universal file format for medical images.
 * Supported voxel types are char, unsigned char, short and unsigned short,
 * int, unsigned int and float.
 */

class irtkFileGIPLToImage : public irtkFileToImage
{

protected:

  /// Read header of GIPL file
  virtual void ReadHeader();

public:

  /// Returns name of class
  virtual const char *NameOfClass();

  /// Returns whether file has correct header
  static int CheckHeader(const char *);
};

#endif
