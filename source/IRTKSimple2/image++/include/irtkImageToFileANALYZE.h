/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageToFileANALYZE.h 8 2009-03-02 16:12:58Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-03-02 16:12:58 +0000 (Mon, 02 Mar 2009) $
  Version   : $Revision: 8 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKIMAGETOFILEANALYZE_H

#define _IRTKIMAGETOFILEANALYZE_H

/**
 * Class for image to ANALYZE file filter.
 *
 * This is a class which takes an image as input and produces an image file
 * in ANALYZE file format.
 */

class irtkImageToFileANALYZE : public irtkImageToFile
{

  /// Filename of header
  char *_headername;

protected:

  /// Initialize filter
  virtual void Initialize();

public:

  /// Constructor
  irtkImageToFileANALYZE();

  /// Destructor
  virtual ~irtkImageToFileANALYZE();

  /// Set input
  virtual void SetOutput(const char *);

  /// Return name of class
  virtual const char *NameOfClass();

};

#endif
