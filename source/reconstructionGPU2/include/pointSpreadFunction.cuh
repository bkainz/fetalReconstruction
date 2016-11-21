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

#include "matrix4.cuh"
#include "reconConfig.cuh"
#include <limits>


template <typename T>
class PointSpreadFunction
{
public: 
  float3 m_PSFdim;      // recon_dim = 0.75
  uint3 m_PSFsize;      // PSF_SIZE = 128
  Matrix4<T> m_PSFI2W;
  Matrix4<T> m_PSFW2I;
  T m_quality_factor;


  __device__ __host__ T sinc_pi(const T x)
  {
    // similar to boost::sinc
    T const    taylor_0_bound = std::numeric_limits<T>::epsilon();
    T const    taylor_2_bound = std::sqrt(taylor_0_bound);
    T const    taylor_n_bound = std::sqrt(taylor_2_bound);

    if (abs(x) >= taylor_n_bound)
    { 
      return(sin(x)/x);

    }else{  // approximation by taylor series in x at 0 up to order 0
      T    result = static_cast<T>(1);
      if (abs(x) >= taylor_0_bound)
      {
          T x2 = x*x;
          // approximation by taylor series in x at 0 up to order 2
          result -= x2/static_cast<T>(6);
          if (abs(x) >= taylor_2_bound)
          {
              // approximation by taylor series in x at 0 up to order 4
              result += (x2*x2)/static_cast<T>(120);
          }
      }
      return(result);
    }
  }


  __device__ __host__ T inline calcPSF(float3 sPos, float3 dim)
  {
    const T sigmaz = dim.z/* / 2.3548f*/;

    #if 1
    #if !USE_SINC_PSF
        const T sigmax = 1.2f * dim.x / 2.3548f;
        const T sigmay = 1.2f * dim.y / 2.3548f;
        return exp((-sPos.x * sPos.x) / (2.0f * sigmax * sigmax) - (sPos.y * sPos.y) / (2.0f * sigmay * sigmay)
          - (sPos.z * sPos.z) / (2.0f * sigmaz * sigmaz));
    #else
        // sinc is already 1.2 FWHM
        sPos.x  = sPos.x * dim.x / 2.3548f;
        sPos.y  = sPos.y * dim.y / 2.3548f;
        T x     = sqrt(sPos.x*sPos.x + sPos.y*sPos.y);
        T R     = 3.14159265359f * x;
        T si    = sinc_pi(R);
        // T si    = sin(R)/R;
        
        return  si*si * exp((-sPos.z * sPos.z) / (2.0f * sigmaz * sigmaz)); //Bartlett positive sinc

    #endif

    #else
        const T sigmax = 1.2f * dim.x / 2.3548f;
        const T sigmay = 1.2f * dim.y / 2.3548f;
        T val = exp(-sPos.x * sPos.x / (2.0f * sigmax * sigmax) - sPos.y * sPos.y / (2.0f * sigmay * sigmay)
          - sPos.z * sPos.z / (2.0f * sigmaz * sigmaz));
        return val;
    #endif
  }

  __device__ __host__ T inline getPSFParamsPrecomp(float3 &ofsPos, const float3& psfxyz, int3 currentoffsetMCenter, Matrix4<T> combInvTrans, float3 patchPos, float3 patchDim)
  {
    ofsPos = make_float3(currentoffsetMCenter.x + psfxyz.x, currentoffsetMCenter.y + psfxyz.y, currentoffsetMCenter.z + psfxyz.z);

    float3 psfxyz2 = combInvTrans*ofsPos;
    float3 psfofs = psfxyz2 - patchPos; //only in patch coordintes we are sure about z
    psfofs = make_float3(psfofs.x*patchDim.x, psfofs.y*patchDim.y, psfofs.z*patchDim.z / 2.5f); //TODO find out why factor required

    psfxyz2 = psfofs - m_PSFI2W*make_float3(((m_PSFsize.x - 1)*0.5f), ((m_PSFsize.y - 1)*0.5f), ((m_PSFsize.z - 1)*0.5f));
    return calcPSF(make_float3(psfxyz2.x, psfxyz2.y, psfxyz2.z), patchDim);
  }

};
