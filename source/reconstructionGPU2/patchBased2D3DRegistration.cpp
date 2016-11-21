/*=========================================================================
* GPU accelerated motion compensation for MRI
*
* Copyright (c) 2016 Bernhard Kainz, Amir Alansary, Maria Kuklisova-Murgasova,
* Kevin Keraudren, Markus Steinberger
* (b.kainz@imperial.ac.uk)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
=========================================================================*/
#include "patchBased2D3DRegistration.h"
#include <irtkImage.h>
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>
#include "patchBasedVolume.cuh"
#include "reconVolume.cuh"
#include "patchBased2D3DRegistration_gpu2.cuh"
//#include "experimentalCuda2D3DReg/patchBasedLBFGS2D3DRegistration_gpu.cuh"
#include "smallPatchesHybridReg/irtkImageRigidRegistrationWithPadding_hybrid.h"

//check if this class is necessary, or if we could use patchBased2D3DRegistration_gpu
template <typename T>
void patchBased2D3DRegistration_gpu2(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction);

template <typename T>
patchBased2D3DRegistration<T>::patchBased2D3DRegistration(int cuda_device, PatchBasedVolume<T>* inputStack, 
	ReconVolume<T>* reconstruction, irtkGenericImage<char>* _mask, T _thickness) :
	m_cuda_device(cuda_device), m_inputStack(inputStack), m_reconstruction(reconstruction), m_mask(_mask), m_thickness(_thickness)
{
  strm_buffer = cout.rdbuf();
  strm_buffer_e = cerr.rdbuf();
  file.open("log-2D3DregCPU.txt");
  file_e.open("log-2D3Dreg-errorCPU.txt");
}

template <typename T>
patchBased2D3DRegistration<T>::~patchBased2D3DRegistration()
{

}

template <typename T>
void patchBased2D3DRegistration<T>::run()
{
  //TODO: block level reduction for even more performance -- no texture used
  PatchBased2D3DRegistration_gpu2<T> registration(m_cuda_device, m_inputStack, m_reconstruction);
  registration.run();
}


template<>
void patchBased2D3DRegistration<float>::runExperimental()
{
  //textures work only for float, which is the reason for explicit implementations here
  //TODO: double registration and better and faster registration
  //patchBasedLBFGS2D3DRegistration_gpu<float> registration(0, m_inputStack, m_reconstruction);
 // registration.run();
}

template<>
void patchBased2D3DRegistration<double>::runExperimental()
{
//TODO
}

/////////////////////////////////////////////////////////////////////////////7
//Test on CPU for comparison

template <typename T>
class ParallelPatchToVolumeRegistration {
public:
  std::vector<irtkGenericImage<T> > _patches;
  std::vector<irtkRigidTransformation>* _transformations;
  irtkGenericImage<T> _CPUreconstruction;

  ParallelPatchToVolumeRegistration(irtkGenericImage<T> reconstruction, std::vector<irtkGenericImage<T> > patches,
    std::vector<irtkRigidTransformation>* transformations) :
    _CPUreconstruction(reconstruction), _patches(patches), _transformations(transformations) {}

  void operator() (const blocked_range<size_t> &r) const {

    irtkImageAttributes attr = _CPUreconstruction.GetImageAttributes();
    //irtkGreyImage source = _CPUreconstruction;
    //irtkImageRigidRegistrationWithPadding registration;
   // registration.SetSource(&source);

    //TODO this uses a lot of GPU memory!
    for (size_t inputIndex = r.begin(); inputIndex != r.end(); ++inputIndex) {

      //irtkGreyPixel smin, smax;
      irtkImageRigidRegistrationWithPadding registration;
      irtkGreyImage source = _CPUreconstruction;
      irtkGreyImage target;
      irtkGreyPixel smin, smax;

      irtkGenericImage<T> patch = _patches[inputIndex];
      if (patch.GetXSize() != attr._dx && patch.GetYSize() != attr._dy)
      {
        // irtkResamplingWithPadding<T> resampling(attr._dx, attr._dx, attr._dx, -1);
        irtkResamplingWithPadding<T> resampling(attr._dx, attr._dy, attr._dz, -1);
        irtkGenericImage<T> patch = _patches[inputIndex];
        resampling.SetInput(&patch);
        resampling.SetOutput(&patch);
        resampling.Run();
      }
      target = patch;

      target.GetMinMax(&smin, &smax);

      if (smax > -1) {
        //put origin to zero
        irtkRigidTransformation offset;
        patchBased2D3DRegistration<T>::ResetOrigin(target, offset);
        irtkMatrix mo = offset.GetMatrix();
        irtkMatrix m = _transformations->at(inputIndex).GetMatrix();
        m = m*mo;
        _transformations->at(inputIndex).PutMatrix(m);
        //std::cout << " ofsMatrix: " << inputIndex << std::endl;
        //reconstructor->_transformations[inputIndex].GetMatrix().Print();

       // registration.SetTarget(&target);

        registration.SetInput(&target, &source);
        registration.SetOutput(&(_transformations->at(inputIndex)));
        registration.GuessParameterSliceToVolume(false);
        registration.SetTargetPadding(-1);
        registration.Run();

        //reconstructor->_slices_regCertainty[inputIndex] = registration.last_similarity;
        //undo the offset
        mo.Invert();
        m = _transformations->at(inputIndex).GetMatrix();
        m = m*mo;
        _transformations->at(inputIndex).PutMatrix(m);
      }

     printf(".");
    }

    //registration.freeGPU();
  }

  // execute
  void operator() () const {
    task_scheduler_init init(tbb_no_threads);
    parallel_for(blocked_range<size_t>(0, _patches.size()), *this);
    init.terminate();
    printf("\n");
  }

};


template <typename T>
void patchBased2D3DRegistration<T>::ResetOrigin(irtkGreyImage &image, irtkRigidTransformation& transformation)
{
  double ox, oy, oz;
  image.GetOrigin(ox, oy, oz);
  image.PutOrigin(0, 0, 0);
  transformation.PutTranslationX(ox);
  transformation.PutTranslationY(oy);
  transformation.PutTranslationZ(oz);
  transformation.PutRotationX(0);
  transformation.PutRotationY(0);
  transformation.PutRotationZ(0);
}

template <typename T>
void patchBased2D3DRegistration<T>::runHybrid()
{
  cerr.rdbuf(file_e.rdbuf());
  cout.rdbuf(file.rdbuf());

  irtkGenericImage<T> stack = m_inputStack->getInputStack();
  irtkImageAttributes attr = m_CPUreconstruction->GetImageAttributes();

  //generate patches CPU // slices
  if (m_patches.empty())
  {
    generatePatchesCPU();
  }

  ParallelPatchToVolumeRegistration<T> registration(*m_CPUreconstruction, m_patches, &m_transformations);
  registration();

  for (unsigned int i = 0; i < m_transformations.size(); i++)
  {
    if (i > 100 && i < 110)
      m_transformations[i].GetMatrix().Print();
  }
  //TODO reactivate!
  //update transformations in m_inputStack for comparison and shortcut
  m_inputStack->updateTransformationMatrices(m_transformations);

  cout.rdbuf(strm_buffer);
  cerr.rdbuf(strm_buffer_e);


  for (unsigned int i = 0; i < m_transformations.size(); i++)
  {
    if (i > 100 && i < 110)
      m_transformations[i].GetMatrix().Print();
  }
}


template <typename T>
void patchBased2D3DRegistration<T>::generatePatchesCPU(bool useFullSlices)
{
	//TODO get directly from pB_volumes

  m_transformations.clear();
  m_patches.clear();

  uint2 pbbsize = m_inputStack->getPbbSize();
  uint2 stride = m_inputStack->getStride();
  irtkGenericImage<T> stack = m_inputStack->getInputStack();
  irtkImageAttributes attr = stack.GetImageAttributes();
  if (useFullSlices)
  {
	  pbbsize.x = attr._x;
	  pbbsize.y = attr._y;
	  stride.x = attr._x + 1;
	  stride.y = attr._y + 1;
  }

  for (int z = 0; z < stack.GetZ(); z++)
  {
    irtkGenericImage<T> slice = stack.GetRegion(0, 0, z, attr._x, attr._y, z + 1);
	slice.PutPixelSize(attr._dx, attr._dy, m_thickness * 2);
    irtkImageAttributes sattr = slice.GetImageAttributes();
    sattr._x = pbbsize.x;
    sattr._y = pbbsize.y;

    for (int y = 0; y < (int)(stack.GetY() + pbbsize.y); y += (int)stride.y)
    {
      for (int x = 0; x < (int)(stack.GetX() + pbbsize.x); x += (int)stride.x)
      {
        ImagePatch2D<T> p;
      /*  p.m_bbsize = pbbsize;
        p.m_pos.x = x;
        p.m_pos.y = y;
        p.m_pos.z = z;*/
        p.scale = 1.0;
        p.patchWeight = 1.0f;

        sattr._xorigin = 0;
        sattr._yorigin = 0;
        sattr._zorigin = 0;
        irtkGenericImage<T> patch(sattr);

        // Calculate position of first voxel in roi in original image
        double x1 = x;
        double y1 = y;
        double z1 = 0;
        slice.ImageToWorld(x1, y1, z1);

        // Calculate position of first voxel in roi in new image
        double x2 = 0;
        double y2 = 0;
        double z2 = 0;
        patch.ImageToWorld(x2, y2, z2);

        // Shift origin of new image accordingly
        patch.PutOrigin(x1 - x2, y1 - y2, z1 - z2);

       /* irtkMatrix trans = m_stackTransformation.GetMatrix();
        p.Transformation = toMatrix4<float>(trans);
        trans.Invert();
        p.InvTransformation = toMatrix4<float>(trans);*/
        //irtkMatrix trans = m_inputStack->m_stackTransformation.GetMatrix();
        irtkRigidTransformation trans = m_inputStack->getStackTransformation();

        int setCount = 0;
        for (int j = 0; j < pbbsize.y; j++)
        {
          for (int i = 0; i < pbbsize.x; i++)
          {
            double xx, yy, zz;
            xx = i;
            yy = j;
            zz = 0;
            patch.ImageToWorld(xx, yy, zz);
            slice.WorldToImage(xx, yy, zz);

            double xx1 = i;
            double yy1 = j;
            double zz1 = 0;
            patch.ImageToWorld(xx1, yy1, zz1);
            m_mask->WorldToImage(xx1, yy1, zz1);

            if (xx >= 0 && yy >= 0 && xx < slice.GetX() && yy < slice.GetY())
            {
              if (xx1 >= 0 && yy1 >= 0 && zz1 >= 0 && xx1 < m_mask->GetX() && yy1 < m_mask->GetY() && zz1 < m_mask->GetZ())
              {
                if (m_mask->Get(xx1, yy1, zz1) != 0)
                {
                  patch(i, j, 0) = slice(xx, yy, 0);
                  if (patch(i, j, 0) != 0 && patch(i, j, 0) != -1) setCount++;
                }
              }
            }
          }
        }

        Matrix4<T> I2W = toMatrix4<T>(patch.GetImageToWorldMatrix());
        Matrix4<T> W2I = toMatrix4<T>(patch.GetWorldToImageMatrix());
        p.I2W = I2W;
        p.W2I = W2I;

        //TODO: There must be a better way
        irtkRigidTransformation offset;
        double ox, oy, oz;
        patch.GetOrigin(ox, oy, oz);
        offset.PutTranslationX(ox);
        offset.PutTranslationY(oy);
        offset.PutTranslationZ(oz);
        offset.PutRotationX(0);
        offset.PutRotationY(0);
        offset.PutRotationZ(0);

        irtkGenericImage<T> patch2(patch);
        patch2.PutOrigin(0, 0, 0);
        p.RI2W = toMatrix4<T>(patch2.GetImageToWorldMatrix());
        irtkMatrix mo = offset.GetMatrix();
        p.Mo = toMatrix4<T>(mo);
        irtkMatrix moinv = mo;
        moinv.Invert();
        p.InvMo = toMatrix4<T>(moinv);

        //origin alone should be enough!?
        //p.m_origin = make_float3(ox, oy, oz);

        /*p.m_dim.x = patch.GetXSize();
        p.m_dim.y = patch.GetYSize();
        p.m_dim.z = patch.GetZSize();*/

		if (setCount > 1.0f / 3.0f * pbbsize.y*pbbsize.x && !useFullSlices)
        {
          //test ok -- will be overwritten by new stack
         /* if (m_patches.size() > 120 && m_patches.size() < 125)
          {
            char buffer[256];
            sprintf(buffer, "testpatchCPU%i.nii", m_patches.size());
            patch.Write(buffer);
          }*/

          m_patches.push_back(patch);
          m_transformations.push_back(trans);
        }

      }
    }
  }

  printf("m_patches CPU size: %d\n", m_patches.size());

}

template class patchBased2D3DRegistration < float >;
template class patchBased2D3DRegistration < double >;
