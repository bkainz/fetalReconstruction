/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkNoise.cc 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#include <sys/types.h>

#ifndef WIN32
#include <sys/time.h>
#endif

#include <irtkImage.h>

#include <irtkNoise.h>

template <class VoxelType> irtkNoise<VoxelType>::irtkNoise(double Amplitude)
{
#ifndef WIN32
  timeval tv;
  gettimeofday(&tv, NULL);
  _Init = tv.tv_usec;
  _Amplitude = Amplitude;
#else
  cerr << "irtkNoise<VoxelType>::irtkNoise: Not implemented yet for Windows" << endl;
  exit(1);
#endif
}

template <class VoxelType> bool irtkNoise<VoxelType>::RequiresBuffering()
{
  return false;
}

template <class VoxelType> const char *irtkNoise<VoxelType>::NameOfClass()
{
  return "irtkNoise";
}

template class irtkNoise<irtkBytePixel>;
template class irtkNoise<irtkGreyPixel>;
template class irtkNoise<irtkRealPixel>;
