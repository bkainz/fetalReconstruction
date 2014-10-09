/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkGradientImageX.cc 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#include <irtkImage.h>

#include <irtkGradientImage.h>

template <class VoxelType> bool irtkGradientImageX<VoxelType>::RequiresBuffering(void)
{
  return true;
}

template <class VoxelType> const char *irtkGradientImageX<VoxelType>::NameOfClass()
{
  return "irtkGradientImageX";
}

template <class VoxelType> double irtkGradientImageX<VoxelType>::Run(int x, int y, int z, int t)
{

  double previous = this->_input->Get(x-1, y, z, t);
  double next     = this->_input->Get(x+1, y, z, t);

  double gradient = previous - next;

  return gradient;
}


template <class VoxelType> void irtkGradientImageX<VoxelType>::Run()
{
  int x, y, z, t;

  // Do the initial set up
  this->Initialize();

  if (this->_input->GetX() < 2) {
    cerr<<" irtkGradientImageX: Dimensions of input image are wrong"<<endl;
    exit(1);
  }

  for ( t = 0; t < this->_input->GetT(); ++t) {
    for ( z = 0; z < this->_input->GetZ(); ++z) {
      for ( y = 0; y < this->_input->GetY(); ++y) {
        this->_output->Put(0, y, z, t, 0);
        for ( x = 1; x < this->_input->GetX()-1; ++x) {
          this->_output->PutAsDouble(x, y, z, t, this->Run(x, y, z, t));
        }
        this->_output->Put(x, y, z, t, 0);
      }
    }
  }

  // Do the final cleaning up
  this->Finalize();

}

template class  irtkGradientImageX<irtkBytePixel>;
template class  irtkGradientImageX<irtkGreyPixel>;
template class  irtkGradientImageX<irtkRealPixel>;
