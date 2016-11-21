/*
#include "patchBasedObject.cuh"

template <typename T>
__host__ void PatchBasedObject<T>::generateSuperpixels(uint2 & pbbsize, uint2 & stride)
{
  int    spx_sz = 0;   // initialize the superpixel size 
  float  noLabels = 2;   // control number of superpixels using-> (int)(noLabels * sqrt( [width * height] /2 )) ;
  double compactness = 10;  // It was used to control the superpixel shape [compactness factor], but now it is redundant as in SLICO it is defined automatically per each superpixel
  Superpixels<T> superpixesl;
  superpixesl.runStackSLIC(m_h_stack, m_h_spx_stack, spx_sz, noLabels, compactness);

  pbbsize = make_uint2(3 * spx_sz, 3 * spx_sz); // In order to fix the patch size for all the superpixels, we define patch size 3 times the uniform superpixel size. The extra free space is used later for dilation
  stride = make_uint2(0.25*spx_sz, 0.25*spx_sz); // define dilation iterations = 0.25 * superpixel_size

  cout << "Patch size: " << pbbsize.x << "x" << pbbsize.y << endl;
  cout << "Dilation iterations: " << stride.x << endl;
}

template<> class PatchBasedObject < float >;
template<> class PatchBasedObject < double >;

*/