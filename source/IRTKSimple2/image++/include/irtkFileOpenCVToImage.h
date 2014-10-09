/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkFileOpenCVToImage.h 440 2011-11-08 22:21:43Z ws207 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2011-11-08 22:21:43 +0000 (Tue, 08 Nov 2011) $
  Version   : $Revision: 440 $
  Changes   : $Author: ws207 $

=========================================================================*/

#ifndef _IRTKFILECVTOIMAGE

#define _IRTKFILECVTOIMAGE

#ifdef HAS_OPENCV
#include <cv.h>
#include <highgui.h>

/**
 * Class for reading images in PGM file format.
 *
 * This is a class which reads images in PGM file format and converts them
 * into images. The PGM (portable graymap) file format is a file format for
 * 2D images and is defined in pgm(1). At the moment only images in PGM raw
 * file format are supported.
 */

class irtkFileOpenCVToImage : public irtkFileToImage
{
protected:

  IplImage* _pimage;

  virtual void ReadHeader();

public:

  /// Return name of class
  virtual const char *NameOfClass();

  /// Return whether file has correct header
  static int CheckHeader(const char *);

  /// Get output
  virtual irtkImage *GetOutput();

  /// Set input
  virtual void SetInput (const char *);
};

#endif

#endif
