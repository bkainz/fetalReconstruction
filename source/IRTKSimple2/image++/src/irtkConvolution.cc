/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkConvolution.cc 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#include <irtkImage.h>

#include <irtkConvolution.h>

template <class VoxelType> irtkConvolution<VoxelType>::irtkConvolution(bool Normalization)
{
  _Normalization = Normalization;
}

template class irtkConvolution<unsigned char>;
template class irtkConvolution<short>;
template class irtkConvolution<unsigned short>;
template class irtkConvolution<float>;
template class irtkConvolution<double>;
