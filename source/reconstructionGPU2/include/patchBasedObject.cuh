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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "reconConfig.cuh"
#include "volume.cuh"
#include "ImagePatch2D.cuh"

//TODO I don't like this here
using namespace std;
#include <irtkVector.h>
#include <irtkMatrix.h>
#include <irtkImage.h>
#include <irtkTransformation.h>

#include <boost/io/ios_state.hpp>

#include <irtkHistogram.h>
#include <irtkDilation.h>
#include "runStackSLIC.h"

template <typename T>
class PatchBasedObject : public Volume<T> {
public:
  //This class is the base class for volumes that are subdived into patches
  //TODO: SLIC extensions

  virtual void init(uint3 s, float3 d, uint2 pbbsize, uint2 stride, irtkGenericImage<char>& _mask, T _thickness)
  {
    m_pbbsize = pbbsize;
    m_stride = stride;
    m_size = s;
    m_dim = d;
    m_mask = _mask;
    m_thickness = _thickness;
  };

  virtual void init(irtkGenericImage<T> & stack, irtkRigidTransformation stacktransformation_, uint2 pbbsize, uint2 stride, T _thickness)
  {
    m_pbbsize = pbbsize;
    m_stride = stride;
    m_size = make_uint3(stack.GetX(), stack.GetY(), stack.GetZ()); //s;
    m_dim = make_float3(stack.GetXSize(), stack.GetYSize(), stack.GetZSize()); //d;
    m_h_stack = stack;
    m_stackTransformation = stacktransformation_;
    m_thickness = _thickness;
  };

  virtual void release(){
    cudaFree(d_patches);
  };

  __device__ __host__ uint3 getSize(){ return m_size; };
  __device__ __host__ float3 getDim(){ return m_dim; };


  virtual __host__  __device__ uint3 getXYZPatchGridSize()
  {
#ifdef __CUDACC__
    return m_XYZPatchGridSize;
#else
    if (m_numPatches == 0)
    {
      printf("WARING: generating standard patch size 64x64 stride: 32x32 \n");
      int total_pixels = 0;
      generate2DPatches(make_uint2(64, 64), make_uint2(32, 32),total_pixels);
    }
    uint3 out;
    out.x = m_pbbsize.x;
    out.y = m_pbbsize.y;
    out.z = m_numPatches;
    m_XYZPatchGridSize = out;
    return out;
#endif
  }

  __host__ __device__ ImagePatch2D<T>* getImagePatch2DPtr()
  {
    return d_patches;
  }

  __device__ ImagePatch2D<T> & getImagePatch2D(const unsigned int & patch)
  {
    return d_patches[patch];
  }

  __device__ void setTransformationMatrix(const unsigned int & patch, const Matrix4<T>& newMatrix)
  {
    d_patches[patch].Transformation = newMatrix;
  }

  __device__ void setInvTransformationMatrix(const unsigned int & patch, const Matrix4<T>& newMatrix)
  {
    d_patches[patch].InvTransformation = newMatrix;
  }

  __host__ irtkGenericImage<T> getInputStack(){ return m_h_stack; };


  uint2 getPbbSize(){ return m_pbbsize; };
  uint2 getStride() { return m_stride; };
  irtkRigidTransformation getStackTransformation() { return m_stackTransformation;};

  // return reconstructed patches for evaluation
  std::vector<ImagePatch2D<T> > getHostImagePatch2DVector(){return _h_patches;};
  std::vector<irtkGenericImage<T> > getHostImagePatchDataVector() {return _h_patchesData;};
  std::vector<irtkGenericImage<T> > getHostImagePatchWeightVector() {return _h_patchesData;};

  void updateHostImagePatch2D()
  {
    std::vector<ImagePatch2D<T> > _h_patches;
    _h_patches.resize(m_numPatches);
    checkCudaErrors(cudaMemcpy(&_h_patches[0], d_patches, m_numPatches*sizeof(ImagePatch2D<T>), cudaMemcpyDeviceToHost));
  }
  // T* h_pointer;
  // malloc(h_pointer, m_size.x*m_size.y*m_size.z*sizeof(T))
  // cudaMemcpy(h_pointer, d_m_PatchesPtr, m_size.x*m_size.y*m_size.z*sizeof(T));

  __host__ void updateTransformationMatrices(std::vector<irtkRigidTransformation> tranfmats)
  {
    if (m_numPatches == 0)
      return;

    std::vector<ImagePatch2D<T> > _h_patches;
    _h_patches.resize(m_numPatches);
    checkCudaErrors(cudaMemcpy(&_h_patches[0], d_patches, m_numPatches*sizeof(ImagePatch2D<T>), cudaMemcpyDeviceToHost));

    for (int i = 0; i < _h_patches.size(); i++)
    {
      irtkMatrix trans = tranfmats[i].GetMatrix();
      _h_patches[i].Transformation = toMatrix4<T>(trans);
      trans.Invert();
      _h_patches[i].InvTransformation = toMatrix4<T>(trans);
    }

    checkCudaErrors(cudaMemcpy(d_patches, &_h_patches[0], m_numPatches*sizeof(ImagePatch2D<T>), cudaMemcpyHostToDevice));
  }


protected:
  //TODO generate slices as patches
  //TODO: do this with mask

  virtual __host__ void generate2DPatches(const uint2 & pbbsize, const uint2 & stride, int &total_pixels, bool useFullSlices = false)
  {
    m_pbbsize = pbbsize;
    m_stride = stride;

    irtkImageAttributes attr      = m_h_stack.GetImageAttributes();
    irtkImageAttributes mask_attr = m_mask.GetImageAttributes();
    if (useFullSlices)
    {
      m_pbbsize.x = attr._x;
      m_pbbsize.y = attr._y;
      m_stride.x = attr._x +1;
      m_stride.y = attr._y + 1;
    }

    printf("m_pbbsize.x %d \n", m_pbbsize.x);
    printf("m_pbbsize.y %d \n", m_pbbsize.y);
    printf("m_stride.x %d \n", m_stride.x);
    printf("m_stride.y %d \n", m_stride.y);

    //2D overlapping patches (boxes)
    //fill patch vector
    //TODO only use these inside mask
    for (int z = 0; z < m_size.z; z++)
    {

      irtkGenericImage<T> slice = m_h_stack.GetRegion(0, 0, z, attr._x, attr._y, z + 1);
      slice.PutPixelSize(attr._dx, attr._dy, m_thickness*2);
      irtkImageAttributes sattr = slice.GetImageAttributes();
      sattr._x = m_pbbsize.x;
      sattr._y = m_pbbsize.y;

      for (int y = 0; y < (int)(m_size.y + m_pbbsize.y); y += (int)m_stride.y)
      {
        for (int x = 0; x < (int)(m_size.x + m_pbbsize.x); x += (int)m_stride.x)
        {
          ImagePatch2D<T> p;
          /*p.m_bbsize = pbbsize;
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

          irtkMatrix trans = m_stackTransformation.GetMatrix();
          p.Transformation = toMatrix4<T>(trans);
          trans.Invert();
          p.InvTransformation = toMatrix4<T>(trans);

          int setCount = 0;
          for (int j = 0; j < m_pbbsize.y; j++)
          {
            for (int i = 0; i < m_pbbsize.x; i++)
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
              m_mask.WorldToImage(xx1, yy1, zz1);

              if (xx >= 0 && yy >= 0 && xx < slice.GetX() && yy < slice.GetY())
              {
                if (xx1 >= 0 && yy1 >= 0 && zz1 >= 0 && xx1 < m_mask.GetX() && yy1 < m_mask.GetY() && zz1 < m_mask.GetZ())
                {
                  if (m_mask.Get(xx1, yy1, zz1) > 0)
                  {
                    patch(i, j, 0) = slice(xx, yy, 0);
                    if (patch(i, j, 0) != 0 && patch(i, j, 0) != -1) setCount++;
                  }
                }
              }
            }
          }

          // // assign the generated patch to the mask
          // for (int j = 0; j < m_pbbsize.y; j++)
          // {
          //   for (int i = 0; i < m_pbbsize.x; i++)
          //   {
          //     if ( patch(i,j,0) != -1 ) p.spxMask[i+96*j] = '1';
          //   }
          // }

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
          patch2.PutOrigin(0,0,0);
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

          if (setCount > 1.0f / 3.0f * m_pbbsize.y*m_pbbsize.x)
          {
            //test ok -- will be overwritten by new stack
            /* if (_h_patches.size() > 120 && _h_patches.size() < 125)
            {
              char buffer[256];
              sprintf(buffer, "testpatchGPU%i.nii", _h_patches.size());
              patch.Write(buffer);
            }*/
            total_pixels += setCount;
            _h_patchesData.push_back(patch);
            _h_patches.push_back(p);
          }

        }
      }
    }

    checkCudaErrors(cudaMalloc((void**)&d_patches, _h_patches.size()*sizeof(ImagePatch2D<T>)));
    checkCudaErrors(cudaMemcpy((d_patches), &_h_patches[0], _h_patches.size()*sizeof(ImagePatch2D<T>), cudaMemcpyHostToDevice));

    m_numPatches = _h_patches.size();
    // cout << m_numPatches << "," ;
    printf("m_patches GPU size: %d ... \n", m_numPatches);
  }


  //TODO: SLIC extensions

  virtual __host__ irtkGenericImage<T> dilatePatch(irtkGenericImage<T> patch)
  {
    for (int j=0; j<patch.GetY(); j++){
      for (int i=0; i<patch.GetX(); i++){
        // cout << "i " <<  i << " j " << j << " patch(i,j,0)  " << patch(i,j,0) << endl;
        if (patch(i,j,0) == 1){
          if (i>0 && patch(i-1,j,0)==0) patch(i-1,j,0) = 2;
          if (j>0 && patch(i,j-1,0)==0) patch(i,j-1,0) = 2;
          if (i+1<patch.GetX() && patch(i+1,j,0)==0) patch(i+1,j,0) = 2;
          if (j+1<patch.GetY() && patch(i,j+1,0)==0) patch(i,j+1,0) = 2;
        }
      }
    }
    for (int j=0; j<patch.GetY(); j++){
      for (int i=0; i<patch.GetX(); i++){
        // cout << "i " <<  i << " j " << j << " patch(i,j,0)  " << patch(i,j,0) << endl;
        if (patch(i,j,0) == 2) patch(i,j,0) = 1;
      }
    }
    return patch;
  }

  void printPatch(irtkGenericImage<T> patch)
  {
    boost::io::ios_all_saver guard(cout); // Saves current flags and format

    for (int j = 0; j < patch.GetY(); j++)
    {
      for (int i = 0; i < patch.GetX(); i++)
      {
        cout << setfill(' ') << setw(5) << (int)patch(i,j,0) <<  "\t";
        // printf("%4.4s ", patch(i,j,0));
      }
      cout << endl;
      // printf("\n");
    }
  }

  virtual __host__ uint2 extractMaxPatchSize()
  {
    uint2 patchSize;
    patchSize.x = 0;
    patchSize.y = 0;
    irtkImageAttributes attr = m_h_stack.GetImageAttributes();

    // extract maximum patch size form all the stacks
    for (int z = 0; z < m_size.z; z++)      // m_size contains stack size -> .x .y .z
    {
      irtkGenericImage<T>   spx_slice = m_h_spx_stack.GetRegion(0, 0, z, attr._x, attr._y, z + 1);
      spx_slice.PutPixelSize(attr._dx, attr._dy, m_thickness*2);

      // get min and max superpixel label
      T minLbl, maxLbl;
      spx_slice.GetMinMax(&minLbl, &maxLbl);

      for (int idxLbl = int(minLbl); idxLbl < int(maxLbl); idxLbl++)
      {
        // initialize region box
        int xMin = INT_MAX, yMin = INT_MAX, xMax = INT_MIN, yMax = INT_MIN;
        bool lblExists = false;
        //extract a boundary box around
        for (int xi =0; xi<spx_slice.GetX(); xi++)
        {
          for (int yi =0; yi<spx_slice.GetY(); yi++)
          {
            if ((int)(spx_slice(xi,yi,0))==idxLbl)
            {
              if (xi<xMin)  xMin = xi;
              if (xi>xMax)  xMax = xi;
              if (yi<yMin)  yMin = yi;
              if (yi>yMax)  yMax = yi;
              lblExists = true;
            }
          }
        }
        if ((xMax - xMin)>patchSize.x) patchSize.x = xMax - xMin;
        if ((yMax - yMin)>patchSize.y) patchSize.y = yMax - yMin;
      }
    }
    return patchSize;
  }

  // This function is very close to generate2DPatches, however, here the main loop is the index of the superpixel label not the x,y first point in the patch.
  // Here xMin and yMin of the patch is defined from extracting the boundary box of the superpixel.
  // And the patch size is around twice the size of the superpixel.
  // The patch is filled with -1 outside the superpixel and the original pixel value inside the superpixel.
  virtual __host__ void generate2DSuperpixelPatches(uint2 &spx_sz, uint2 extendSpx, unsigned int stackNo, int &total_pixels, bool _debug = false)
  {
    // generateSuperpixels
    irtkImageAttributes attr = m_h_stack.GetImageAttributes();
    // check if spxSize is greater than the image size
    if (spx_sz.x>attr._x) { printf("WARNING: spxSize.x is bigger than imageSize.x - force spxSize.x = 0.5 * imageSize.x \n"); spx_sz.x=attr._x/2; }
    if (spx_sz.y>attr._y) { printf("WARNING: spxSize.y is bigger than imageSize.y - force spxSize.y = 0.5 * imageSize.y \n"); spx_sz.y=attr._y/2; }

    runStackSLIC<T> superpixels;
    superpixels.segmentSLIC(m_h_stack, m_h_spx_stack, spx_sz.x, spx_sz.y, stackNo, _debug);
    // char savename[256];
    // sprintf(savename, "test_spx.nii.gz");
    // m_h_spx_stack.Write(savename);
    // std::vector<ImagePatch2D<T> > _h_patches;

    // initialize dilation variables
    float dilateRatio = (float)extendSpx.x/100.;

    // extract patch around superpixels
    m_pbbsize   = extractMaxPatchSize();
    printf("Max patch size - before dilation - %d x %d \n", m_pbbsize.x, m_pbbsize.y);
    printf("Dilation ratio: %d %% \n", extendSpx.x);


    // define fixed patch sizes - TODO: better implementation
    m_pbbsize.x=64;
    m_pbbsize.y=64;

    // m_pbbsize.x = m_pbbsize.x + 2*dilateRatio*extendSpx.x;
    // m_pbbsize.y = m_pbbsize.y + 2*dilateRatio*extendSpx.y;
    // // crop patch if it is bigger than spxMask allowed sizes used in GPU patches
    // if ((m_pbbsize.x<32) || (m_pbbsize.y<32))
    // {
    //   m_pbbsize.x=32;
    //   m_pbbsize.y=32;
    // }
    // else if ((m_pbbsize.x<64) || (m_pbbsize.y<64))
    // {
    //   m_pbbsize.x=64;
    //   m_pbbsize.y=64;
    // }
    // else
    // {
    //   m_pbbsize.x=128;
    //   m_pbbsize.y=128;
    // }

    // check again (after dilation) if it is bigger than the input image dimensions
    if (m_pbbsize.x>attr._x) m_pbbsize.x=attr._x;
    if (m_pbbsize.y>attr._y) m_pbbsize.y=attr._y;

    printf("Max patch size - after dilation - %d x %d \n", m_pbbsize.x, m_pbbsize.y);

    // define local variables per each superpixel
    uint2 m_extend;
    uint2 spxPatchSize;
    int diter;


    // loop superpixels - extend size - dilate - store
    for (int z = 0; z < m_size.z; z++)      // m_size contains stack size -> .x .y .z
    {
      // extract original image slice
      irtkGenericImage<T>   slice     = m_h_stack.GetRegion(0, 0, z, attr._x, attr._y, z + 1);
      slice.PutPixelSize(attr._dx, attr._dy, m_thickness*2);
      // extract labeled superpixel slice
      irtkGenericImage<T>   spx_slice = m_h_spx_stack.GetRegion(0, 0, z, attr._x, attr._y, z + 1);
      spx_slice.PutPixelSize(attr._dx, attr._dy, m_thickness*2);
      // extract slice attributes (dimensions)
      irtkImageAttributes   sattr     = slice.GetImageAttributes();
      sattr._x = m_pbbsize.x;
      sattr._y = m_pbbsize.y;

      // get min and max superpixel label
      T minLbl, maxLbl;
      spx_slice.GetMinMax(&minLbl, &maxLbl);

      // create patches from superpixels
      for (int idxLbl = int(minLbl); idxLbl < int(maxLbl); idxLbl++)
      {
        // initialize region box
        int xMin = INT_MAX, yMin = INT_MAX, xMax = INT_MIN, yMax = INT_MIN;
        bool lblExists = false;

        //extract a boundary box around the current superpixel (idxLbl)
        for (int yi =0; yi<spx_slice.GetY(); yi++)
        {
          for (int xi =0; xi<spx_slice.GetX(); xi++)
          {
            if ((int)(spx_slice(xi,yi,0))==idxLbl)
            {
              if (xi<xMin) xMin = xi;
              if (xi>xMax) xMax = xi;
              if (yi<yMin) yMin = yi;
              if (yi>yMax) yMax = yi;
              lblExists = true;
            }
          }
        }

        // continue if there is no superpixel or a very small superpixel found
        // if (!lblExists ||  ( (xMax - xMin)*(yMax - yMin) < 0.5*(spx_sz.x*spx_sz.y) ) ) continue;
        if (!lblExists) continue;

        // cout << "before -- " <<  "xMin "  << xMin << " xMax " << xMaxPutOrigin << " yMin " << yMin <<  " yMax " << yMax << endl ;

        // original superpixel size
        spxPatchSize.x = xMax-xMin;
        spxPatchSize.y = yMax-yMin;
        // calculate dilation iterations as a ratio of the bigger dimension
        if (spxPatchSize.x > spxPatchSize.y)
        {
          diter = dilateRatio * spxPatchSize.x;
        }else {
          diter = dilateRatio * spxPatchSize.y;
        }
        // calculate extend superpixel size = maxSize-origSize
        m_extend.x  = round( ((float)m_pbbsize.x - (float) spxPatchSize.x) /2.);
        m_extend.y  = round( ((float)m_pbbsize.y - (float) spxPatchSize.y) /2.);

        // extend the superpixel window  (max patchSize)
        if ((int)(xMin - m_extend.x) < 0)
        {
          xMax = m_pbbsize.x;
          xMin = 0;
        }
        else if ((int)(xMax + m_extend.x) > slice.GetX())
        {
          xMax = slice.GetX();
          xMin = xMax - m_pbbsize.x;
        }
        else
        {
          xMin -= m_extend.x;
          xMax  = xMin + m_pbbsize.x;
        }
        if ((int)(yMin - m_extend.y) < 0)
        {
          yMax = m_pbbsize.y;
          yMin = 0;
        }
        else if ((int)(yMax + m_extend.y) > slice.GetY())
        {
          yMax = slice.GetY();
          yMin = yMax - m_pbbsize.y;
        }
        else
        {
          yMin -= m_extend.y;
          yMax  = yMin +  m_pbbsize.y;
        }

        // cout << "after  -- "        << "xMin " << xMin  << " xMax " << xMax << " yMin " << yMin <<  " yMax " << yMax << endl ;
        // cout << "spxPatchSize = "   << spxPatchSize.x   << " x "    << spxPatchSize.y  << endl;
        // cout << "m_extend = "       << m_extend.x       << " x "    << m_extend.y      << endl;
        // cout << "patchSize = "      << (xMax-xMin)      << " x "    << (yMax-yMin)     << endl;

        ImagePatch2D<T> p;
        p.scale = 1.0;
        p.patchWeight = 1.0f;

        // Calculate position of first voxel in roi in original image
        double x1 = xMin;
        double y1 = yMin;
        double z1 = 0;
        slice.ImageToWorld(x1, y1, z1);

        sattr._xorigin = 0;
        sattr._yorigin = 0;
        sattr._zorigin = 0;

        // irtkGenericImage<T> patch(sattr);
        irtkGenericImage<T> patch =  m_h_stack.GetRegion(xMin, yMin, z, xMax, yMax, z + 1);
        patch.PutPixelSize(attr._dx, attr._dy, m_thickness*2);

        double xx = xMin;
        double yy = yMin;
        double zz = 0;
        spx_slice.ImageToWorld(xx, yy, zz);
        patch.WorldToImage(xx, yy, zz);
        // cout << "xx = " << (int)xx << " yy = " << (int)yy << " zz = " << (int)zz << endl;

        irtkMatrix trans = m_stackTransformation.GetMatrix();
        p.Transformation = toMatrix4<T>(trans);
        trans.Invert();
        p.InvTransformation = toMatrix4<T>(trans);

        int setCount = 0;
        for (int j = 0; j < m_pbbsize.y; j++)
        {
          for (int i = 0; i < m_pbbsize.x; i++)
          {
            patch(i, j, 0) == 0;

            double xx, yy, zz;
            xx = i;
            yy = j;
            zz = 0;
            patch.ImageToWorld(xx, yy, zz);
            spx_slice.WorldToImage(xx, yy, zz);

            xx = round(xx);
            yy = round(yy);
            zz = round(zz);

            double xx1, yy1, zz1;
            xx1 = i;
            yy1 = j;
            zz1 = 0;
            patch.ImageToWorld(xx1, yy1, zz1);
            m_mask.WorldToImage(xx1, yy1, zz1);

            xx1 = round(xx1);
            yy1 = round(yy1);
            zz1 = round(zz1);

            // generate a patch mask form superpixels
            patch(i, j, 0) = 0; // patch value = -1 by default

            if (xx >= 0 && yy >= 0 && xx < spx_slice.GetX() && yy < spx_slice.GetY())
            {
              if (xx1 >= 0 && yy1 >= 0 && zz1 >= 0 && xx1 < m_mask.GetX() && yy1 < m_mask.GetY() && zz1 < m_mask.GetZ())
              {
                if (m_mask.Get(xx1, yy1, zz1) > 0) // check it is not outside the cropped mask
                  patch(i, j, 0) = (spx_slice(xx, yy, 0) == idxLbl) ? 1 : 0;  // assign pixel value inside the current superpixel and -1 outside
              }
            }

            // if (patch(i, j, 0) != 0 && patch(i, j, 0) != -1) setCount++;
            if ( patch(i, j, 0) > 0 ) setCount++;
            // cout << "idxLbl = " << idxLbl << " spx_slice(xx, yy, 0) = " << spx_slice(xx, yy, 0) << " patch(i, j, 0) = " << patch(i, j, 0) << endl;

          }
        }

        // ignore small superpixels smaller than half initial superpixel size
        if (setCount < 2 ) continue;
        if (setCount < 1.0f / 4.0f * spx_sz.y*spx_sz.x) continue;


        // // print patch for testing
        // cout << "initial patch "  << z << endl;
        // printPatch(patch);

        // dilate patch mask
        for (int ii = 0; ii < diter; ii++) {
          patch = dilatePatch(patch);
        }

        // // print patch for testing
        // cout << "dilated patch "  << z << endl;
        // printPatch(patch);

        // fill patch mask with original image values
        for (int j = 0; j < m_pbbsize.y; j++)
        {
          for (int i = 0; i < m_pbbsize.x; i++)
          {

            if (patch(i, j, 0) == 0) {
              patch(i, j, 0) = -1;
              continue;
            }

            double xx, yy, zz;
            xx = i;
            yy = j;
            zz = 0;
            patch.ImageToWorld(xx, yy, zz);
            slice.WorldToImage(xx, yy, zz);

            xx = round(xx);
            yy = round(yy);
            zz = round(zz);

            double xx1 = i;
            double yy1 = j;
            double zz1 = 0;
            patch.ImageToWorld(xx1, yy1, zz1);
            m_mask.WorldToImage(xx1, yy1, zz1);

            xx1 = round(xx1);
            yy1 = round(yy1);
            zz1 = round(zz1);

            // generate a patch mask from superpixels
            if (xx >= 0 && yy >= 0 && xx < spx_slice.GetX() && yy < spx_slice.GetY())
            {
              if (xx1 >= 0 && yy1 >= 0 && zz1 >= 0 && xx1 < m_mask.GetX() && yy1 < m_mask.GetY() && zz1 < m_mask.GetZ())
              {
                if (m_mask.Get(xx1, yy1, zz1) > 0) // check it is not outside the cropped mask
                  patch(i, j, 0) = (patch(i, j, 0) == 1) ? slice(xx,yy,0) : -1;  // assign pixel value inside the current superpixel and -1 outside
                else
                  patch(i, j, 0) = -1;
                // cout << "idxLbl = " << idxLbl << " spx_slice(xx, yy, 0) = " << spx_slice(xx, yy, 0) << " patch(i, j, 0) = " << patch(i, j, 0) << endl;
                // if (patch(i, j, 0) != 0 && patch(i, j, 0) != -1) setCount++;
              }
            }

          }
        }
        // assign the generated patch to the mask
        for (int j = 0; j < m_pbbsize.y; j++)
        {
          for (int i = 0; i < m_pbbsize.x; i++)
          {
            if ( patch(i,j,0) != -1 ) {
              p.spxMask[i+64*j] = '1';
              total_pixels      += 1;
            }
          }
        }

        // // print patch for testing
        // cout << "final patch "  << z << endl;
        // printPatch(patch);

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
        patch2.PutOrigin(0,0,0);
        p.RI2W = toMatrix4<T>(patch2.GetImageToWorldMatrix());
        irtkMatrix mo = offset.GetMatrix();
        p.Mo = toMatrix4<T>(mo);
        irtkMatrix moinv = mo;
        moinv.Invert();
        p.InvMo = toMatrix4<T>(moinv);

        // if (setCount > 1.0f / 3.0f * m_pbbsize.y*m_pbbsize.x)
        {
          //test ok -- will be overwritten by new stack
          if (_h_patches.size() > 20 && _h_patches.size() < 25)
          {
            char buffer[256];
            sprintf(buffer, "testpatchGPU%i.nii", _h_patches.size());
            patch.Write(buffer);
          }
        }

        // store patches with their data
        _h_patchesData.push_back(patch);
        _h_patches.push_back(p);

        /*if (setCount > 1.0f / 5.0f * m_pbbsize.y*m_pbbsize.x)
        {
          _h_patches.push_back(p);
        }*/
      }
    }

    checkCudaErrors(cudaMalloc((void**)&d_patches, _h_patches.size()*sizeof(ImagePatch2D<T>)));
    checkCudaErrors(cudaMemcpy((d_patches), &_h_patches[0], _h_patches.size()*sizeof(ImagePatch2D<T>), cudaMemcpyHostToDevice));

    m_numPatches = _h_patches.size();
    // cout << m_numPatches << "," ; // un-comment for patch-extraction
    printf("m_patches GPU size: %d ... \n", m_numPatches);
  }


  uint3 m_XYZPatchGridSize;

  unsigned int m_numPatches;
  uint2 m_pbbsize;
  uint2 m_stride;

  ImagePatch2D<T>* d_patches;
  // TODO stack mask and masked patching
  irtkGenericImage<T> m_h_stack;
  irtkRigidTransformation m_stackTransformation;
  irtkGenericImage<char> m_mask;
  T m_thickness;
  std::vector<ImagePatch2D<T> > _h_patches;
  std::vector<irtkGenericImage<T> > _h_patchesData;

  int   m_extend;
  irtkGenericImage<T> m_h_spx_stack;

  //why are they not inherited when using gcc? stupid c++ standard in gcc
  using Volume<T>::m_size;
  using Volume<T>::m_dim;
  using Volume<T>::m_d_data;

};
