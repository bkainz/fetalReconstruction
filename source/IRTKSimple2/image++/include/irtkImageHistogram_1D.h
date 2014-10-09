/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageHistogram_1D.h 552 2012-02-15 15:18:09Z ws207 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2012-02-15 15:18:09 +0000 (Wed, 15 Feb 2012) $
  Version   : $Revision: 552 $
  Changes   : $Author: ws207 $

=========================================================================*/

#ifndef _IRTKIMAGEHISTOGRAM_1D_H

#define _IRTKIMAGEHISTOGRAM_1D_H

/** Class for 2D histograms.
 *  This class defines and implements 2D histograms.
 */

template <class VoxelType> class irtkImageHistogram_1D : public irtkHistogram_1D<double>
{
protected:
    /// min for equalize
    VoxelType _emin;
    /// max for equalize
    VoxelType _emax;

public:
	/// Evaluate the histogram from a given image with padding value
	virtual void Evaluate(irtkGenericImage<VoxelType> *, double padding = -10000);
	/// Histogram Equalization
	virtual void Equalize(VoxelType min,VoxelType max);
	/// Back project the equalized histogram to image
	virtual void BackProject(irtkGenericImage<VoxelType> *); 
};

#endif
