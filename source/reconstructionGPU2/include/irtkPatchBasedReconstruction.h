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
#include <irtkImage.h>
#include <irtkTransformation.h>
#include <irtkGaussianBlurring.h>
#include <cuda_runtime_api.h>
#include "patchBasedVolume.cuh"
#include "reconVolume.cuh"
#include "pointSpreadFunction.cuh"

template <typename T>
class irtkPatchBasedReconstruction : public irtkObject
{
public:
  irtkPatchBasedReconstruction(int cuda_device, float isoVoxelSize, uint2 patchSize, uint2 patchStride, 
    int iterations = 7, int rec_iterations = 7, int _dilateMask = 0, bool _resample = false, bool _noMatchIntensities = false, bool _superpixel = false, bool _hierarchical = false, bool _debug = false, bool patch_extraction = false);
  ~irtkPatchBasedReconstruction();

  void setImageStacks(const std::vector < irtkGenericImage<T> > & stacks, vector<T> thickness, const vector<string> & inputTransformations = vector<string>(), const vector<int> packageDenominator = vector<int>());
  void setMask(irtkGenericImage<char> & mask);
  void setEvaluationBaseline(bool evaluateBaseline = false);
  void setEvaluationMaskName(vector<string> & evaluationMaskNames);
  void setEvaluationGtName(string & evaluationGtName);
  void run();
  irtkGenericImage<T>* getReconstruction();

  void setExistingReconstructionTarget(irtkGenericImage<T>* target);
  void setPackageDenominators(vector<int> packageDenominators){ m_packageDenominators = packageDenominators; };
  
  void release();

  void Evaluate2d(int iter, string evaluationMaskName);
  void Evaluate3d(int iter, irtkGenericImage<T> reconimage, string evaluationMaskName);
  void EvaluateBaseline2d(string evaluationMaskName);
  void EvaluateBaseline3d(string evaluationMaskName);
  void EvaluateGt3d(int iter, irtkGenericImage<T> reconimage);

  void setUseFullSlices(bool value = false){ useFullSlices = value; };

protected:
  irtkGenericImage<T> CreateMaskFromOverlap(vector < irtkGenericImage<T> > & stacks);
  void TransformMask(irtkGenericImage<T> & image, irtkGenericImage<char> & mask,
    irtkRigidTransformation& transformation);
  void CropImage(irtkGenericImage<T>& image, irtkGenericImage<char>& mask);
  void CreateTemplate(irtkGenericImage<T>& stack, float resolution);
  void computeMinMaxIntensities();
  void writeDebugImages();

  void MatchStackIntensitiesWithMasking(bool together = false);

private:
  vector< irtkGenericImage<T> > m_stacks;
  vector<irtkRigidTransformation> m_stack_transformations;
  irtkGenericImage<char> m_mask;
  // irtkGenericImage<char> m_evaluationMask;
  string m_evaluationGtName;
  vector<string> m_evaluationMaskNames;
  irtkGenericImage<T>* m_reconstruction;
  std::vector<PatchBasedVolume<T> > m_pbVolumes;
  ReconVolume<T> m_GPURecon;
  uint2 m_patchSize;
  uint2 m_patchStride;
  float m_isoVoxelSize;
  int m_dilateMask;
  bool readyForRecon;
  bool haveMask;
  bool haveEvaluationMask;
  bool evaluateGt;
  bool m_evaluateBaseline;
  T m_min_intensity;
  T m_max_intensity;
  bool m_debug;
  bool m_patch_extraction;
  bool m_superpixel;
  bool m_hierarchical;
  bool m_noMatchIntensities;
  bool m_resample;
  T m_average_value;
  std::vector<T> m_stack_factor;
  int m_rec_iterations;
  int m_iterations;
  int m_template_num;
  bool haveExistingReconstructionTarget;
  vector<int> m_packageDenominators;
  vector<T> m_thickness;
  bool useFullSlices;
  int m_cuda_device;
};