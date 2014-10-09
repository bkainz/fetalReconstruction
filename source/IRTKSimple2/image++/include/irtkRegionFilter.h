/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkRegionFilter.h 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef IRTKREGIONFILTER_H

#define IRTKREGIONFILTER_H

#include <irtkImageToImage2.h>

/**
 * Class for extracting regions for images
 *
 */

class irtkRegionFilter : public irtkImageToImage2
{

protected:

	/// Region of interest
	int _i1, _j1, _k1, _l1, _i2, _j2, _k2, _l2; 
	
  /// Returns whether the filter requires buffering
  virtual bool RequiresBuffering();

  /// Returns the name of the class
  virtual const char *NameOfClass();

public:

  /// Constructor
	irtkRegionFilter();

  /// Destructor
  ~irtkRegionFilter();

  /// Define ROI
  virtual void PutRegion(int, int, int, int, int, int, int, int);
  
  /// Extract ROI
  virtual void Run();

};

inline void irtkRegionFilter::PutRegion(int i1, int j1, int k1, int l1, int i2, int j2, int k2, int l2)
{
	_i1 = i1;
	_j1 = j1;
	_k1 = k1;
	_l1 = l1;
	_i2 = i2;
	_j2 = j2;
	_k2 = k2;
	_l2 = l2;
}

#endif 
