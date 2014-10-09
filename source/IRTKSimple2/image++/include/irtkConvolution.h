/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkConvolution.h 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKCONVOLUTION_H

#define _IRTKCONVOLUTION_H

#include <irtkImageToImage.h>

template <class VoxelType> class irtkConvolution : public irtkImageToImage<VoxelType>
{

protected:

  /// Flag whether to normalize convolution
  bool _Normalization;

public:

  /// Constructor
  irtkConvolution(bool = false);

  /// Set normalization on/off
  SetMacro(Normalization, bool);

  /// Set normalization on/off
  GetMacro(Normalization, bool);

};

// Convolution filters without padding
#include <irtkConvolution_1D.h>
#include <irtkConvolution_2D.h>
#include <irtkConvolution_3D.h>

// Convolution filters with padding
#include <irtkConvolutionWithPadding_1D.h>
#include <irtkConvolutionWithPadding_2D.h>
#include <irtkConvolutionWithPadding_3D.h>

#endif
