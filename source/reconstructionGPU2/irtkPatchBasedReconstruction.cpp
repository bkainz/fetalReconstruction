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
#include <irtkPatchBasedReconstruction.h>
#include <irtkResampling.h>
#include <irtkRegistration.h>
#include <irtkImageRigidRegistration.h>
#include <irtkImageRigidRegistrationWithPadding.h>
#include <irtkImageFunction.h>
#include <irtkTransformation.h>
#include <math.h>
#include <stdlib.h>
#include <irtkDilation.h>

#include <irtkImage.h>
#include <irtkHistogram.h>

#include "patchBasedLayeredSurface3D.cuh"
#include "irtkStack3D3DRegistration.h"
#include "patchBased2D3DRegistration.h"
#include "patchBasedSuperresolution_gpu.cuh"
#include "patchBasedRobustStatistics_gpu.cuh"
#include "patchBasedPackageSplitter.h"

#include <perfstats.h>

#include <boost/io/ios_state.hpp>

template <typename T>
void patchBasedPSFReconstruction_gpu(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction, bool useSpx);
template <typename T>
void patchBasedSimulatePatches_gpu(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction);
template <typename T>
void initPatchBasedRecon_gpu(int cuda_device, PatchBasedVolume<T> & inputStack, ReconVolume<T> & reconstruction, PointSpreadFunction<float> & _PSF, bool useSpx);

template <typename T>
irtkPatchBasedReconstruction<T>::irtkPatchBasedReconstruction(int cuda_device, float isoVoxelSize, uint2 patchSize, uint2 patchStride,
    int iterations, int rec_iterations, int _dilateMask, bool _resample, bool _noMatchIntensities, bool _superpixel, bool _hierarchical, bool _debug, bool patch_extraction) :
        m_cuda_device(cuda_device), m_isoVoxelSize(isoVoxelSize), m_reconstruction(NULL), readyForRecon(false), haveMask(false), m_dilateMask(_dilateMask), m_resample(_resample), m_noMatchIntensities(_noMatchIntensities), m_patchSize(patchSize), m_patchStride(patchStride), m_debug(_debug), m_rec_iterations(rec_iterations), m_iterations(iterations), m_superpixel(_superpixel), m_hierarchical(_hierarchical), evaluateGt(false), m_patch_extraction(patch_extraction)

{
    m_reconstruction = new irtkGenericImage<T>(1, 1, 1);
    haveExistingReconstructionTarget = false;
    useFullSlices = false;
}

template <typename T>
irtkPatchBasedReconstruction<T>::~irtkPatchBasedReconstruction()
{
	printf("disconstruct reconstruction and free memory \n");
    for (int i = 0; i < m_pbVolumes.size(); i++)
    {
    	printf("reset stack %i \n", i);
    	m_pbVolumes[i].reset();
    	printf("release stack %i \n", i);
        m_pbVolumes[i].release();
    }
}

template <typename T>
void irtkPatchBasedReconstruction<T>::release()
{
	printf("release reconstruction and free memory \n");

    for (int i = 0; i < m_pbVolumes.size(); i++)
    {
    	printf("reset stack %i \n", i);
    	m_pbVolumes[i].reset();
    	printf("release stack %i \n", i);
        m_pbVolumes[i].release();
    }
}

template <typename T>
void irtkPatchBasedReconstruction<T>::setImageStacks(const std::vector < irtkGenericImage<T> > & stacks, vector<T> thickness, const vector<string> & inputTransformations, const vector<int> packageDenominator)
{
    m_stacks = stacks;
    readyForRecon = true;

    m_template_num = -1;
    for (int i = 0; i < stacks.size(); i++)
    {
        irtkTransformation *transformation;
        if (!inputTransformations.empty())
        {
        	printf("Reading transformations\n");
            if (inputTransformations[i] == std::string("id"))
            {
            	printf("ref stack = %d \n", i);
		        printf("transformation number %d = %s \n",i,(char*)(inputTransformations[i].c_str()));
                if (m_template_num < 0) m_template_num = i;
                transformation = new irtkRigidTransformation;
            }
            else
            {
            	printf("transformation number %d = %s \n",i,(char*)(inputTransformations[i].c_str()));
                transformation = irtkTransformation::New((char*)(inputTransformations[i].c_str()));
            }
        }
        else
        {
            transformation = new irtkRigidTransformation;
            if (m_template_num < 0) m_template_num = 0;
        }

        irtkRigidTransformation *rigidTransf = dynamic_cast<irtkRigidTransformation*> (transformation);
        m_stack_transformations.push_back(*rigidTransf);
        delete rigidTransf;
    }


    //transform stack vector to vector of package volumes if packageDenominator is given
    m_packageDenominators = packageDenominator;
    m_thickness = thickness;
    if (packageDenominator.size() == stacks.size())
    {
        //TODO improve
        printf("splitting volumes into Packages...\n");
        patchBasedPackageSplitter<T> packageSplitter(m_template_num, &m_stacks, &m_stack_transformations, thickness, packageDenominator);
        m_stacks = packageSplitter.getPackageVolumes();
        m_stack_transformations = packageSplitter.getPackageTransformations();
        m_thickness = packageSplitter.getPackageThickness();
    }
}

template <typename T>
void irtkPatchBasedReconstruction<T>::setMask(irtkGenericImage<char> & mask)
{
    m_mask = mask;
    haveMask = true;
}

template <typename T>
void irtkPatchBasedReconstruction<T>::setEvaluationBaseline(bool evaluateBaseline)
{
	m_evaluateBaseline = evaluateBaseline;
}

template <typename T>
void irtkPatchBasedReconstruction<T>::setEvaluationMaskName(vector<string> & evaluationMaskNames)
{
    m_evaluationMaskNames = evaluationMaskNames;
    haveEvaluationMask = true;
}

template <typename T>
void irtkPatchBasedReconstruction<T>::setEvaluationGtName(string & evaluationGtName)
{
    m_evaluationGtName = evaluationGtName;
    evaluateGt = true;
}

template <typename T>
irtkGenericImage<T>* irtkPatchBasedReconstruction<T>::getReconstruction()
{
    if (!readyForRecon)
    {
        printf("ERROR: no input set\n");
    }
    return m_reconstruction;
}

template <typename T>
void irtkPatchBasedReconstruction<T>::setExistingReconstructionTarget(irtkGenericImage<T>* target)
{
    haveExistingReconstructionTarget = true;
    m_reconstruction = target;
    readyForRecon = true;
}

template <typename T>
void irtkPatchBasedReconstruction<T>::run()
{
  // create mask from overlapping if there is no mask given
  if (!haveMask)
  {
      m_mask   = CreateMaskFromOverlap(m_stacks);
      haveMask = true;
  }else{ // fix mask values
      for (int k = m_mask.GetZ() - 1; k >= 0; k--) {
          for (int j = m_mask.GetY() - 1; j >= 0; j--) {
              for (int i = m_mask.GetX() - 1; i >= 0; i--) {
                  if ((unsigned int)m_mask.Get(i, j, k) == 0){ m_mask(i, j, k)=0; }else{ m_mask(i, j, k)=1;}
              }
          }
      }
  }

    // dilate mask to include boundry information
	if (m_dilateMask)
	{
		printf("Dilate reconstruction mask %d-iterations \n", m_dilateMask);
		irtkDilation<T> dilation;
		irtkGenericImage<T> m_tmp = m_mask;
		dilation.SetConnectivity(CONNECTIVITY_26);
		dilation.SetInput(&m_tmp);
		dilation.SetOutput(&m_tmp);
		for (int i = 0; i < m_dilateMask; i++) dilation.Run();
		m_mask = m_tmp;
	}

  // resample
  // irtkNearestNeighborInterpolateImageFunction interpolator;
  irtkBSplineInterpolateImageFunction interpolatorGrey;
  irtkResampling<T> resampling(m_isoVoxelSize, m_isoVoxelSize, m_isoVoxelSize);

  for (int i = 0; i < m_stacks.size(); i++)
  {
    irtkGenericImage<char> m = m_mask;
    TransformMask(m_stacks[i], m, m_stack_transformations[i]);

    //Crop stack
    CropImage(m_stacks[i], m);

    if (m_resample)
		{
			// resample stack before reocnstruction for a better quality, however, it consumes a lot of memory on GPU!
			printf("Resample input stack %d \n", i);
	        resampling.SetInput(&m_stacks[i]);
	        resampling.SetOutput(&m_stacks[i]);
	        resampling.SetInterpolator(&interpolatorGrey);
	        resampling.Run();
		}

      if (m_debug)
      {
          char buffer[256];
          sprintf(buffer, "stack%i.nii.gz", i);
          m_stacks[i].Write(buffer);
      }
  }


  // resample mask
  // if (m_resample)
	{
		irtkGenericImage<T> uint_mask = m_mask;
		irtkNearestNeighborInterpolateImageFunction interpolatorBinary;
		resampling.SetInput(&uint_mask);
		resampling.SetOutput(&uint_mask);
		resampling.SetInterpolator(&interpolatorBinary);
		resampling.Run();
		m_mask = uint_mask;
	}

    if (m_debug)
    {
        m_mask.Write("mask.nii.gz");
    }

    // claculate the max and min intensities from all input stacks
    computeMinMaxIntensities();

    // baseline evaluation using first and last stack
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;
    if (m_evaluateBaseline) { for (int j = 0; j < m_evaluationMaskNames.size(); j++) { EvaluateBaseline3d(m_evaluationMaskNames[j]); } }
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;

    //////////////////////////////////////
    //3D-3D registration if desired
    //this is modular to be able to replace it with a better registration
    //////////////////////////////////////
    irtkStack3D3DRegistration<T> stackRegistrator(m_template_num, &m_stacks, &m_stack_transformations, &m_mask);
    stackRegistrator.run();

    if (!m_noMatchIntensities)
    {
        printf("Intensities before -- min: %f max: %f \n", m_min_intensity, m_max_intensity);
	    MatchStackIntensitiesWithMasking();
	    computeMinMaxIntensities();
		printf("After min: %f max: %f \n", m_min_intensity, m_max_intensity);
	}

    if (!haveExistingReconstructionTarget)
    {
        CreateTemplate(m_stacks[m_template_num], m_isoVoxelSize);
    }

    //TODO Gaussfilter would make it nicer
    irtkGenericImage<char> reconmask = m_mask;
    // irtkRigidTransformation transformation;
    TransformMask(*m_reconstruction, reconmask, m_stack_transformations[m_template_num]);
    // reconmask.Write("testtmask.nii");
    cout << "m_cuda_device " << m_cuda_device << endl;
    m_GPURecon.init(m_cuda_device, make_uint3(m_reconstruction->GetX(), m_reconstruction->GetY(), m_reconstruction->GetZ()), make_float3(m_reconstruction->GetXSize(), m_reconstruction->GetYSize(), m_reconstruction->GetZSize()), toMatrix4<float>(m_reconstruction->GetWorldToImageMatrix()), toMatrix4<float>(m_reconstruction->GetImageToWorldMatrix()));
    m_GPURecon.setMask(reconmask.GetPointerToVoxels());
    
    if (haveExistingReconstructionTarget)
    {
        //copy to device
        m_GPURecon.copyFromHost(m_reconstruction->GetPointerToVoxels());
    }

    // stream buffer memory
	char buffer[256];

	// evaluate patch extraction
	if (m_patch_extraction) 
	{
		streambuf* strm_buffer = cout.rdbuf();

		std::string tag;
		if (m_superpixel)
		{
			tag = (m_hierarchical) ? "hspx" : "spx";
		}else{
			tag = (m_hierarchical) ? "hpatch" : "patch";
		}
	    
	    sprintf(buffer, "patch-extraction-%s-size-%i-stride%i.csv", tag.c_str(), m_patchSize.x, m_patchStride.x);
	    ofstream file(buffer); // ofstream file(buffer, std::ios_base::app);
	    cout.rdbuf(file.rdbuf()); // assign streambuf to cout

	    // prepare the evaluation file
	    cout << "iter" 			  << ",";
	    cout << "stack_no"        << ",";
		cout << "patch_size"      << ",";
		cout << "patch_stride"    << ",";
		cout << "num_patches" 	  << ",";
		cout << "total_pixels"    << ",";
		cout << "stack_pixels"    << ",";
		cout << "patch_pixels"    << ",";
		cout << "overhead_pixels" << ",";
		cout << "overhead_ratio"  << ",";
		cout << endl;

		printf("number of iterations = %d \n", m_iterations);

		pt::ptime 	patches_tick = pt::microsec_clock::local_time();

    	for (int iter = 0; iter <= m_iterations; iter++)
	    {	
	    	for (int i = 0; i < m_stacks.size(); i++)
		    {
		     	cout << iter << ",";
		     	cout << i << ",";
		        cout << m_patchSize.x << ",";
		        cout << m_patchStride.x  << ",";

		        printf("Thickness %f \n", m_thickness[i]);

		        PatchBasedVolume<T> pBStack;
		        pBStack.init(m_stacks[i], m_stack_transformations[i], m_patchSize, m_patchStride, m_mask, m_thickness[i], m_superpixel, useFullSlices, m_debug, m_patch_extraction, i);
		        pBStack.copyFromHost(m_stacks[i].GetPointerToVoxels());
		        // m_pbVolumes.push_back(pBStack);
		    }

		    if (m_hierarchical) { 
		    	m_patchSize.x   += 8; m_patchSize.y   += 8;
                if ( !m_superpixel) { m_patchStride.x += 4; m_patchStride.y += 4;}
            }

		}

	   	// restore cout's original streambuf and close stream files
	    cout.rdbuf(strm_buffer);
		
		pt::ptime 			now 	= pt::microsec_clock::local_time();
	    pt::time_duration 	diff 	= now - patches_tick;
	    double 				mss 	= diff.total_milliseconds() / 1000.0;

	    cout << "patch_extraction_time" << "," << mss << endl;
		return;

	}
	

    // extract patches
	for (int i = 0; i < m_stacks.size(); i++)
    {
    	cout << "stack [" << i << "] -------------------------- " << endl;
        printf("Thickness %f \n", m_thickness[i]);
        PatchBasedVolume<T> pBStack;
        pBStack.init(m_stacks[i], m_stack_transformations[i], m_patchSize, m_patchStride, m_mask, m_thickness[i], m_superpixel, useFullSlices, m_debug, m_patch_extraction, i);
        pBStack.copyFromHost(m_stacks[i].GetPointerToVoxels());
        m_pbVolumes.push_back(pBStack);
    }

 
    //TODO multi GPU
    irtkImageAttributes attr;
    attr._x  = PSF_SIZE;
    attr._y  = PSF_SIZE;
    attr._z  = PSF_SIZE;
    attr._dx = m_isoVoxelSize;
    attr._dy = m_isoVoxelSize;
    attr._dz = m_isoVoxelSize;
    irtkGenericImage<T> PSFimg(attr);

    PointSpreadFunction<float> h_PSF;
    h_PSF.m_PSFI2W 	= toMatrix4<float>(PSFimg.GetImageToWorldMatrix());
    h_PSF.m_PSFW2I 	= toMatrix4<float>(PSFimg.GetWorldToImageMatrix());
    h_PSF.m_quality_factor = 1.0f;
    h_PSF.m_PSFsize = make_uint3(attr._x, attr._y, attr._z);
    h_PSF.m_PSFdim 	= make_float3(attr._dx, attr._dy, attr._dz);

    // char buffer[256];
    //superresolution and RS has to be global object (scale, and EM parameters)
    //TODO make registrators similar in usage
    patchBasedSuperresolution_gpu<T> superresolution(m_min_intensity, m_max_intensity);
    patchBasedRobustStatistics_gpu<T> robustStatistics(m_pbVolumes);

    //Extract patches and their transformations on GPU
    for (int i = 0; i < m_pbVolumes.size(); i++)
    {
        //For each stack on one GPU!
        initPatchBasedRecon_gpu(m_cuda_device, m_pbVolumes[i], m_GPURecon, h_PSF, m_superpixel);
    }

    std::vector<patchBased2D3DRegistration<T>*> patchRegistrators;
    for (int i = 0; i < m_pbVolumes.size(); i++)
    {
        patchBased2D3DRegistration<T>* reg = new patchBased2D3DRegistration<T>(m_cuda_device, &m_pbVolumes[i], &m_GPURecon, &m_mask, m_thickness[i]);
        if (useFullSlices)
        {
            reg->generatePatchesCPU();
        }
        patchRegistrators.push_back(reg);
    }

    m_GPURecon.checkGPUMemory();

    for (int iter = 0; iter <= m_iterations; iter++)
    {
    	pt::ptime recon_tick = pt::microsec_clock::local_time();

    	if (useFullSlices)
    	{
    		printf("----------------------------------- Iteration number [%i] -- useFullSlices ------------------------------------- \n", iter);
    	}else{
            printf("----------------------------------- Iteration number [%i] -- patch size (%i,%i) -------------------------------- \n", iter, m_patchSize.x, m_patchStride.x);
        }

        if (iter > 0 || haveExistingReconstructionTarget)
        {
            //TODO test:
            m_GPURecon.copyToHost(m_reconstruction->GetPointerToVoxels());
            for (int i = 0; i < m_pbVolumes.size(); i++)
            {
                pt::ptime tick = pt::microsec_clock::local_time();
                patchRegistrators[i]->m_CPUreconstruction = m_reconstruction;

                if (useFullSlices)
                {
                    printf("registration patches of stack %d hybrid CPU-GPU\n", i);
                    patchRegistrators[i]->runHybrid();
                }
                else
                {
                    printf("registration patches of stack %d \n", i);
                    // TODO: needs integration of treereduction
                    //patchRegistrators[i]->run();
                    //CPU reg for now
                    patchRegistrators[i]->runHybrid();
                }

                pt::ptime now = pt::microsec_clock::local_time();
                pt::time_duration diff = now - tick;
                double mss = diff.total_milliseconds() / 1000.0;

                printf("registration took %f s\n", mss);
            }
            //TODO test end
            writeDebugImages();
            // std::cin.get();
        }

        robustStatistics.initializeEMValues();

        m_GPURecon.reset();
        for (int i = 0; i < m_pbVolumes.size(); i++)
        {
            patchBasedPSFReconstruction_gpu(m_cuda_device, m_pbVolumes[i], m_GPURecon, m_superpixel);
        }

/*		irtkGenericImage<T> reconimage_test(m_reconstruction->GetImageAttributes());
		checkCudaErrors(cudaMemcpy(reconimage_test.GetPointerToVoxels(), m_GPURecon.getDataPtr(), reconimage_test.GetX()*reconimage_test.GetY()*reconimage_test.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
		sprintf(buffer, "reconimage_after_psf.nii.gz");
		reconimage_test.Write(buffer);
*/
        //needs to equalize only once for all stacks TODO: make this clear
        m_GPURecon.equalize();

/*		checkCudaErrors(cudaMemcpy(reconimage_test.GetPointerToVoxels(), m_GPURecon.getDataPtr(), reconimage_test.GetX()*reconimage_test.GetY()*reconimage_test.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
		sprintf(buffer, "reconimage_after_equalize.nii.gz");
		reconimage_test.Write(buffer);*/

        for (int i = 0; i < m_pbVolumes.size(); i++)
        {
            patchBasedSimulatePatches_gpu(m_cuda_device, m_pbVolumes[i], m_GPURecon);
        }

/*		checkCudaErrors(cudaMemcpy(reconimage_test.GetPointerToVoxels(), m_GPURecon.getDataPtr(), reconimage_test.GetX()*reconimage_test.GetY()*reconimage_test.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
		sprintf(buffer, "reconimage_after_simulate_patches.nii.gz");
		reconimage_test.Write(buffer);*/

        robustStatistics.InitializeRobustStatistics(m_min_intensity, m_max_intensity, m_cuda_device);
        robustStatistics.EStep();

		pt::ptime sr_tick = pt::microsec_clock::local_time();

        for (int i = 0; i < m_rec_iterations; i++)
        {
            robustStatistics.Scale();

            m_GPURecon.resetAddonCmap();
            for (int i = 0; i < m_pbVolumes.size(); i++)
            {
                superresolution.run(m_cuda_device, &m_pbVolumes[i], &m_GPURecon);
            }

            superresolution.regularize(m_cuda_device, &m_GPURecon);

            for (int i = 0; i < m_pbVolumes.size(); i++)
            {
                patchBasedSimulatePatches_gpu(m_cuda_device, m_pbVolumes[i], m_GPURecon);
            }

            robustStatistics.MStep(i + 1);
            robustStatistics.EStep();

            writeDebugImages();
        }

        // now 	= pt::microsec_clock::local_time();
        // diff 	= now - sr_tick;
        // mss 	= diff.total_milliseconds() / 1000.0;
        // printf("superresolution took %f s\n", mss);

        // diff 	= now - recon_tick;
        // mss 	= diff.total_milliseconds() / 1000.0;
        // printf("reconstruction took %f s\n", mss);


        // if (m_debug)
        // {
            irtkGenericImage<T> reconimage(m_reconstruction->GetImageAttributes());
            checkCudaErrors(cudaMemcpy(reconimage.GetPointerToVoxels(), m_GPURecon.getDataPtr(), reconimage.GetX()*reconimage.GetY()*reconimage.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
            if (useFullSlices)
        	{
        		sprintf(buffer, "reconimage%i_useFullSlices.nii.gz", iter);
        	}else{
        		sprintf(buffer, "reconimage%i_%i_%i.nii.gz", iter, m_patchSize.x, m_patchStride.x);
        	}
            reconimage.Write(buffer);
        // }

        // evaluate the reconstruction for the given evaluation masks
        cout << "----------------------------------------------------------------------------------------------------------------" << endl;
        if (haveEvaluationMask)
        {
            for (int j = 0; j < m_evaluationMaskNames.size(); j++) { Evaluate3d(iter, reconimage, m_evaluationMaskNames[j]);}
        }

        if (evaluateGt)
        {
            EvaluateGt3d(iter, reconimage);
        }

    /* if (iter > 0)
    {
      std::cin.get();
    }*/

    }

    //RestoreSliceIntensitiesGPU()
    //ScaleVolume();

    m_GPURecon.copyToHost(m_reconstruction->GetPointerToVoxels());
}

template <typename T>
void irtkPatchBasedReconstruction<T>::writeDebugImages()
{
    if (m_debug)
    {
        m_GPURecon.checkGPUMemory();
        char buffer[256];

        for (int i = 0; i < m_pbVolumes.size(); i++)
        {
            unsigned int N = m_pbVolumes[i].getXYZPatchGridSize().x* m_pbVolumes[i].getXYZPatchGridSize().y* m_pbVolumes[i].getXYZPatchGridSize().z;
            irtkGenericImage<T> img(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(img.GetPointerToVoxels(), m_pbVolumes[i].getPatchesPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
            sprintf(buffer, "patches%i.nii.gz", i);
            img.Write(buffer);

            irtkGenericImage<T> imgR(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(imgR.GetPointerToVoxels(), m_pbVolumes[i].getRegPatchesPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
            sprintf(buffer, "fromReconRegPatches%i.nii.gz", i);
            imgR.Write(buffer);

            irtkGenericImage<T> imgB(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(imgB.GetPointerToVoxels(), m_pbVolumes[i].getBufferPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
            sprintf(buffer, "fromInputRegPatches%i.nii.gz", i);
            imgB.Write(buffer);

            irtkGenericImage<char> imgC(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(imgC.GetPointerToVoxels(), m_pbVolumes[i].getSimInsidePtr(), N*sizeof(char), cudaMemcpyDeviceToHost));
            sprintf(buffer, "SimInside%i.nii.gz", i);
            imgC.Write(buffer);

            irtkGenericImage<T> imgD(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(imgD.GetPointerToVoxels(), m_pbVolumes[i].getSimWeightsPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
            sprintf(buffer, "SimWeights%i.nii.gz", i);
            imgD.Write(buffer);

            irtkGenericImage<T> imgE(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(imgE.GetPointerToVoxels(), m_pbVolumes[i].getSimPatchesPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
            sprintf(buffer, "SimPatches%i.nii.gz", i);
            imgE.Write(buffer);

            irtkGenericImage<T> imgF(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
            checkCudaErrors(cudaMemcpy(imgF.GetPointerToVoxels(), m_pbVolumes[i].getWeigthDataPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
            sprintf(buffer, "Weights%i.nii.gz", i);
            imgF.Write(buffer);
        }

        irtkGenericImage<T> addon(m_reconstruction->GetImageAttributes());
        irtkGenericImage<T> cmap(m_reconstruction->GetImageAttributes());
        checkCudaErrors(cudaMemcpy(addon.GetPointerToVoxels(), m_GPURecon.getAddonPtr(), addon.GetX()*addon.GetY()*addon.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(cmap.GetPointerToVoxels(), m_GPURecon.getCMapPtr(), cmap.GetX()*cmap.GetY()*cmap.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
        sprintf(buffer, "Addon%i.nii.gz", 0);
        addon.Write(buffer);
        sprintf(buffer, "cmap%i.nii.gz", 0);
        cmap.Write(buffer);
        ///DEBUG end
    }
}


template <typename T>
void irtkPatchBasedReconstruction<T>::MatchStackIntensitiesWithMasking(bool together)
{
    if (m_debug)
        cout << "Matching intensities of stacks. ";

    //calculate real average value
    m_average_value = 0;
    unsigned long long int count = 0;
    for (int i = 0; i < m_stacks.size(); i++) {
        for (int j = 0; j < m_stacks[i].GetNumberOfVoxels(); j++) {
            if (m_stacks[i].GetPointerToVoxels()[j]<=0) continue;
            m_average_value += m_stacks[i].GetPointerToVoxels()[j];
            count++;
        }
    }
    if (count != 0)
        m_average_value /= count;

    printf("Average value: %f \n", m_average_value);

    //Calculate the averages of intensities for all stacks
    double sum, num;
    char buffer[256];
    unsigned int ind;
    int i, j, k;
    double x, y, z;
    vector<double> stack_average;
    irtkRealImage m;

    //remember the set average value
    //m_average_value = averageValue;

    //averages need to be calculated only in ROI
    for (ind = 0; ind < m_stacks.size(); ind++) {
        m 	= m_stacks[ind];
        sum = 0;
        num = 0;

        for (i = 0; i < m_stacks[ind].GetX(); i++)
            for (j = 0; j < m_stacks[ind].GetY(); j++)
                for (k = 0; k < m_stacks[ind].GetZ(); k++) {
                    //image coordinates of the stack voxel
                    x = i;
                    y = j;
                    z = k;
                    //change to world coordinates
                    m_stacks[ind].ImageToWorld(x, y, z);
                    //transform to template (and also _mask) space
                    m_stack_transformations[ind].Transform(x, y, z);
                    //change to mask image coordinates - mask is aligned with template
                    m_mask.WorldToImage(x, y, z);
                    x = round(x);
                    y = round(y);
                    z = round(z);
                    //if the voxel is inside mask ROI include it
                    if ((x >= 0) && (x < m_mask.GetX()) && (y >= 0) && (y < m_mask.GetY()) && (z >= 0) && (z < m_mask.GetZ()))
                    {
                        if (m_mask(x, y, z) == 1)
                        {
                            m(i, j, k) = 1;
                            if (m_stacks[ind](i, j, k)>0) {
                                sum += m_stacks[ind](i, j, k);
                            	num++;
                            }
                        }
                        else
                            m(i, j, k) = 0;
                    }
                    else
                        m(i, j, k) = 0;
                }
        if (m_debug)
        {
            sprintf(buffer, "mask-for-matching%i.nii.gz", ind);
            m.Write(buffer);
        }
        //calculate average for the stack
        if (num > 0){
            stack_average.push_back(sum / num);
        } else {
            cerr << "Stack " << ind << " has no overlap with ROI" << endl;
            exit(1);
        }
    }

    double global_average;
    if (together) {
        global_average = 0;
        for (i = 0; i < stack_average.size(); i++)
            global_average += stack_average[i];
        global_average /= stack_average.size();
    }

    if (m_debug) {
        cout << "Stack average intensities are ";
        for (ind = 0; ind < stack_average.size(); ind++)
            cout << stack_average[ind] << " ";
        cout << endl;
    }

    //Rescale stacks
    T *ptr;
    double factor;
    for (ind = 0; ind < m_stacks.size(); ind++) {
        if (together) {
            factor = m_average_value / global_average;
            m_stack_factor.push_back((T)factor);
        } else {
            factor = m_average_value / stack_average[ind];
            m_stack_factor.push_back((T)factor);
        }

        ptr = m_stacks[ind].GetPointerToVoxels();
        for (i = 0; i < m_stacks[ind].GetNumberOfVoxels(); i++) {
            if (*ptr > 0)
                *ptr *= factor;
            ptr++;
        }
    }

    if (m_debug) {
        for (ind = 0; ind < m_stacks.size(); ind++) {
            sprintf(buffer, "rescaled-stack%i.nii.gz", ind);
            m_stacks[ind].Write(buffer);
        }

        cout << "Patch intensity factors are ";
        for (ind = 0; ind < stack_average.size(); ind++)
            cout << m_stack_factor[ind] << " ";
        cout << endl;
        cout << "The new average value is " << global_average << endl;
    }

}

template <typename T>
void irtkPatchBasedReconstruction<T>::computeMinMaxIntensities()
{
    T _max_intensity = voxel_limits<T>::min();
    T _min_intensity = voxel_limits<T>::max();

    for (int i = 0; i < m_stacks.size(); i++)
    {
        //to update minimum we need to exclude padding value
        T *ptr = m_stacks[i].GetPointerToVoxels();
        for (int ind = 0; ind < m_stacks[i].GetNumberOfVoxels(); ind++) {
            if (*ptr > 0) {
                if (*ptr > _max_intensity)
                    _max_intensity = *ptr;
                if (*ptr < _min_intensity)
                    _min_intensity = *ptr;
            }
            ptr++;
        }
    }

    m_max_intensity = _max_intensity;
    m_min_intensity = _min_intensity;
}

template <typename T>
void irtkPatchBasedReconstruction<T>::TransformMask(irtkGenericImage<T> & image, irtkGenericImage<char> & mask, irtkRigidTransformation& transformation)
{
    //transform mask to the space of image
    irtkImageTransformation imagetransformation;
    irtkNearestNeighborInterpolateImageFunction interpolator;
    imagetransformation.SetInput(&mask, &transformation);
    irtkGenericImage<T> m = image;
    imagetransformation.SetOutput(&m);
    //target contains zeros and ones image, need padding -1
    imagetransformation.PutTargetPaddingValue(-1);
    //need to fill voxels in target where there is no info from source with zeroes
    imagetransformation.PutSourcePaddingValue(0);
    imagetransformation.PutInterpolator(&interpolator);
    imagetransformation.Run();
    mask = m;
}


template <typename T>
void irtkPatchBasedReconstruction<T>::CropImage(irtkGenericImage<T>& image, irtkGenericImage<char>& mask)
{
    //Crops the image according to the mask
    int i, j, k;
    //ROI boundaries
    int x1, x2, y1, y2, z1, z2;

    //Original ROI
    x1 = 0;
    y1 = 0;
    z1 = 0;
    x2 = image.GetX();
    y2 = image.GetY();
    z2 = image.GetZ();

    //upper boundary for z coordinate
    int sum = 0;
    for (k = image.GetZ() - 1; k >= 0; k--) {
        sum = 0;
        for (j = image.GetY() - 1; j >= 0; j--)
            for (i = image.GetX() - 1; i >= 0; i--)
                if ((uint)mask.Get(i, j, k) > 0)
                    sum++;
        if (sum > 0)
            break;
    }
    z2 = k;

    //lower boundary for z coordinate
    sum = 0;
    for (k = 0; k <= image.GetZ() - 1; k++) {
        sum = 0;
        for (j = image.GetY() - 1; j >= 0; j--)
            for (i = image.GetX() - 1; i >= 0; i--)
                if ((uint)mask.Get(i, j, k) > 0)
                    sum++;
        if (sum > 0)
            break;
    }
    z1 = k;

    //upper boundary for y coordinate
    sum = 0;
    for (j = image.GetY() - 1; j >= 0; j--) {
        sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--)
            for (i = image.GetX() - 1; i >= 0; i--)
                if ((uint)mask.Get(i, j, k) > 0)
                    sum++;
        if (sum > 0)
            break;
    }
    y2 = j;

    //lower boundary for y coordinate
    sum = 0;
    for (j = 0; j <= image.GetY() - 1; j++) {
        sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--)
            for (i = image.GetX() - 1; i >= 0; i--)
                if ((uint)mask.Get(i, j, k) > 0)
                    sum++;
        if (sum > 0)
            break;
    }
    y1 = j;

    //upper boundary for x coordinate
    sum = 0;
    for (i = image.GetX() - 1; i >= 0; i--) {
        sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--)
            for (j = image.GetY() - 1; j >= 0; j--)
                if ((uint)mask.Get(i, j, k) > 0)
                    sum++;
        if (sum > 0)
            break;
    }
    x2 = i;

    //lower boundary for x coordinate
    sum = 0;
    for (i = 0; i <= image.GetX() - 1; i++) {
        sum = 0;
        for (k = image.GetZ() - 1; k >= 0; k--)
            for (j = image.GetY() - 1; j >= 0; j--)
                if ((uint)mask.Get(i, j, k) > 0)
                    sum++;
        if (sum > 0)
            break;
    }

    x1 = i;

    //  if (_debug)
    //    cout << "Region of interest is " << x1 << " " << y1 << " " << z1 << " " << x2 << " " << y2
    //   << " " << z2 << endl;

    // //Cut region of interest
    // cout << image.GetX() << " " << image.GetY() << " " << image.GetZ() << endl;
    // cout << x1 << " " << y1 << " " << z1 << " " << x2 + 1 << " " << y2 + 1 << " " << z2 + 1 << " " << endl;
    image = image.GetRegion(x1, y1, z1, x2 + 1, y2 + 1, z2 + 1);
}

template <typename T>
void irtkPatchBasedReconstruction<T>::CreateTemplate(irtkGenericImage<T>& stack, float resolution)
{
    double dx, dy, dz, d;

    //Get image attributes - image size and voxel size
    irtkImageAttributes attr = stack.GetImageAttributes();

    //create recon image
    m_reconstruction = new irtkGenericImage<T>(attr);
    d = resolution;

    cout << "Constructing volume with isotropic voxel size " << d << " mm." << endl;

    //resample "recon" to resolution "d"
    // irtkNearestNeighborInterpolateImageFunction interpolator;
    irtkBSplineInterpolateImageFunction interpolator;
    irtkResampling<T> resampling(d, d, d);
    resampling.SetInput(m_reconstruction);
    resampling.SetOutput(m_reconstruction);
    resampling.SetInterpolator(&interpolator);
    resampling.Run();

    readyForRecon = true;
}

template <typename T>
irtkGenericImage<T> irtkPatchBasedReconstruction<T>::CreateMaskFromOverlap(std::vector < irtkGenericImage<T> > & stacks)
{
    std::cout << "creating mask from overlap " << std::endl;
    //calculated the non-zero union of the input stacks

    irtkGenericImage<T> avImage = stacks[0];
    avImage = 0;

    for (int z = 0; z < avImage.GetZ(); z++)
    {
        for (int y = 0; y < avImage.GetY(); y++)
        {
            for (int x = 0; x < avImage.GetX(); x++)
            {
                double xx = x;
                double yy = y;
                double zz = z;

                avImage.ImageToWorld(xx, yy, zz);
                bool inside = true;
                for (int i = 0; i < stacks.size(); i++)
                {
                    double xx1 = xx;
                    double yy1 = yy;
                    double zz1 = zz;
                    stacks[i].WorldToImage(xx1, yy1, zz1);
                    if (!(xx1 >= 0 && yy1 >= 0 && zz1 >= 0 && xx1 < stacks[i].GetX() && yy1 < stacks[i].GetY() && zz1 < stacks[i].GetZ()))
                    {
                        inside = false;
                    }
                }
                if (inside) avImage(x, y, z) = 1;

            }
        }
    }

    return avImage;

}

#define DEFAULT_BINS 255

template <typename T>
void irtkPatchBasedReconstruction<T>::EvaluateBaseline2d(string evaluationMaskName)
{
    size_t found = evaluationMaskName.find("brain_1");
    string name_mask = "brain_1";
    if (found == -1) {found = evaluationMaskName.find("brain_2");   name_mask = "brain_2";};
    if (found == -1) {found = evaluationMaskName.find("placenta");  name_mask = "placenta";};
    if (found == -1) {found = evaluationMaskName.find("uterus");    name_mask = "uterus";};
    if (found == -1) {found = 0;};

    // size_t found = evaluationMaskName.find("placenta");
    // string name_mask = "placenta";
    // if (found == -1) {return;};

    printf("Evaluate baseline for %.*s ", 8, name_mask.c_str());

    // define reference and target stacks
    int i = 0; // reference stack
    int target_stack = m_pbVolumes.size()-1;

    // read the evaluation mask from disk
    irtkGenericImage<T> evaluationMask;
    evaluationMask.Read((char*)(evaluationMaskName.c_str()));
    // resample mask
    irtkResampling<T> resampling(m_isoVoxelSize, m_isoVoxelSize, m_isoVoxelSize);
    irtkNearestNeighborInterpolateImageFunction interpolatorBinary;
    resampling.SetInput(&evaluationMask);
    resampling.SetOutput(&evaluationMask);
    resampling.SetInterpolator(&interpolatorBinary);
    resampling.Run();

    irtkGenericImage<T> refStack = m_stacks[i];
    // unsigned int N = m_pbVolumes[i].getXYZPatchGridSize().x* m_pbVolumes[i].getXYZPatchGridSize().y* m_pbVolumes[i].getXYZPatchGridSize().z;
    // irtkGenericImage<T> refStack(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
    // checkCudaErrors(cudaMemcpy(refStack.GetPointerToVoxels(), m_pbVolumes[i].getPatchesPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));

    // printf("Ref image size = (%i,%i,%i)\n",refStack.GetX(),refStack.GetY(),refStack.GetZ());
    // printf("Mask image size = (%i,%i,%i)\n",evaluationMask.GetX(),evaluationMask.GetY(),evaluationMask.GetZ());

    // variables initialization
    char buffer[256];
    double widthx, widthy;

    // read W2I matrices for mask and reference stacks
    Matrix4<float> maskW2I =  toMatrix4<float>(evaluationMask.GetWorldToImageMatrix());
    Matrix4<float> refW2I  =  toMatrix4<float>(refStack.GetWorldToImageMatrix());


    int nbins_x = 0, nbins_y = 0;   // Default number of bins for histogram

    // sprintf(buffer, "evaluateReconimage.nii.gz");
    // reconimage.Write(buffer);

    // sprintf(buffer, "evaluateMaskimage.nii.gz");
    // evaluationMask.Write(buffer);

    cout << "using stacks " <<  i << " and " <<  target_stack << " ... ";

    // stream buffer memory
    streambuf* strm_buffer = cout.rdbuf();
    sprintf(buffer, "log-evaluate-stack-%i-%i-baseline-size-%i-%i-%.*s.csv", i, target_stack, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str()); // construct output stream files
    ofstream file(buffer);
    cout.rdbuf(file.rdbuf()); // assign streambuf to cout


    // prepare the evaluation file
    cout << "Stack[" << i << "]" << "//Patch no."   << ",";
    cout << "PSNR"                                  << ",";
    cout << "SSIM"                                  << ",";
    cout << "DSSIM"                                 << ",";
    cout << "PatchMean"                             << ",";
    cout << "ReconMean"                             << ",";
    cout << "PatchVariance"                         << ",";
    cout << "ReconVariance"                         << ",";
    cout << "Covariance"                            << ",";
    cout << "JointEntropy"                          << ",";
    cout << "Crosscorrelation"                      << ",";
    cout << "CorrelationRatioPatchRecon"            << ",";
    cout << "CorrelationRatioReconPatch"            << ",";
    cout << "MutualInformation"                     << ",";
    cout << "NormalizedMutualInformation"           << ",";
    cout << "SumSquareDiff"                         << ",";
    cout << "LabelConsistency"                      << ",";
    cout << "KappaStatistic"                        << ",";
    cout << endl;

    // update pb_volumes with the generated patches from the GPU
    m_pbVolumes[target_stack].updateHostImagePatch2D();
    std::vector<ImagePatch2D<T> >     patchesTrans = m_pbVolumes[target_stack].getHostImagePatch2DVector();
    std::vector<irtkGenericImage<T> > patchesData  = m_pbVolumes[target_stack].getHostImagePatchDataVector();

    // Set min and max of histogram ----------------------------------------------------------------------------------------------------------------------
    // Calculate number of bins to use
    if (nbins_x == 0) {
        nbins_x = (int) round(m_max_intensity - m_min_intensity) + 1;
        if (nbins_x > DEFAULT_BINS)
            nbins_x = DEFAULT_BINS;
    }
    if (nbins_y == 0) {
        nbins_y = (int) round(m_max_intensity - m_min_intensity) + 1;
        if (nbins_y > DEFAULT_BINS)
            nbins_y = DEFAULT_BINS;
    }

    // SSIM default settings
    // two variables to stabilize the division with weak denominator;
    // L the dynamic range of the pixel-values (typically this is 2^{\#bits_per_pixel}-1);
    // scriptstyle k_1 = 0.01 and  k_2 = 0.03 by default
    // for 255 color range C1 = 6.5025, C2 = 58.5225;
    double C1 = 6.5025, C2 = 58.5225;

    for (unsigned int z = 0; z < patchesData.size(); z++)
    {
        // Create histogram ----------------------------------------------------------------------------------------------------------------------------------------
        irtkHistogram_2D<int> histogram(nbins_x, nbins_y);
        widthx = (m_max_intensity - m_min_intensity) / (nbins_x - 1.0);
        widthy = (m_max_intensity - m_min_intensity) / (nbins_y - 1.0);

        histogram.PutMin(m_min_intensity - 0.5*widthx, m_min_intensity - 0.5*widthy);
        histogram.PutMax(m_max_intensity + 0.5*widthx, m_max_intensity + 0.5*widthy);


        ImagePatch2D<T>     patchTrans = patchesTrans[z];
        irtkGenericImage<T> patchData  = patchesData[z];

       // printPatch(patchData);
        double  sum       = 0;
        uint    numVoxels = 0;

        // Fill histogram by looping all points in the patch --------------------------------------------------------------------------------------------------
        for (unsigned int y = 0; y < patchData.GetY(); y++) {
            for (unsigned int x = 0; x < patchData.GetX(); x++) {

                // patch value
                float3 patchPosF = make_float3(x, y, 0);
                uint3 patchPosU = make_uint3(x, y, 0);
                T patchVal = patchData(patchPosU.x , patchPosU.y , patchPosU.z);
                if (patchVal<=0) continue;   //skip if there is no values in the mask image at this location

                // reference stack value
                float3 _refPos = refW2I*  (patchTrans.Transformation * (patchTrans.I2W * patchPosF));
                uint3  refPos  = make_uint3(round_(_refPos.x), round_(_refPos.y), round_(_refPos.z));

                // mask space position
                float3 _maskPos = maskW2I* (patchTrans.Transformation *(patchTrans.I2W * patchPosF));
                uint3  maskPos  = make_uint3(round_(_maskPos.x), round_(_maskPos.y), round_(_maskPos.z));

                // char maskVal = evaluationMask(maskPos.x, maskPos.y, maskPos.z);
                // printf("maskVal = %u \n", maskVal);

                // printf("patchPos(%i,%i,%i)\n", patchPosU.x , patchPosU.y , patchPosU.z);
                // printf("refPos(%i,%i,%i)\n", refPos.x , refPos.y , refPos.z);
                // printf("maskPos(%i,%i,%i)\n", maskPos.x , maskPos.y , maskPos.z);

                 if (patchVal<=0) continue; //skip if there is no values in the patch image at this location

                // check if out of mask and FoV
                if ((maskPos.x >= 0) && (maskPos.x < evaluationMask.GetX()) && (maskPos.y >= 0) && (maskPos.y < evaluationMask.GetY()) && (maskPos.z >= 0) && (maskPos.z < evaluationMask.GetZ())) {
                    if ((refPos.x >= 0) && (refPos.x < refStack.GetX()) && (refPos.y >= 0) && (refPos.y < refStack.GetY()) && (refPos.z >= 0) && (refPos.z < refStack.GetZ())) {

                        char maskVal = evaluationMask(maskPos.x, maskPos.y, maskPos.z);
                        if (maskVal<=0) continue;   //skip if there is no values in the mask image at this location

                        T refVal = refStack(refPos.x, refPos.y, refPos.z);
                        if (refVal<=0) continue;   //skip if there is no values in the mask image at this location

                        // printf("maskVal = %u patchVal = %f and refVal = %f \n", maskVal, patchVal, refVal);

                        histogram.AddSample(patchVal, refVal);
                        sum += (patchVal - refVal)*(patchVal - refVal);
                        numVoxels +=1;

                    }
                }

            }
        } // end of voxels loop in one patch

        // check if there are voxels in the patch, if not assign zeros for all evaluation metrics
        if (numVoxels<2){continue;}

        sum = sum/numVoxels;
        sum = 20*log10(m_max_intensity) - 10*log10(sum);

        double SSIM = ((2*histogram.MeanX()*histogram.MeanY()+C1)*(2*histogram.Covariance()+C2))
                        / ( (pow(histogram.MeanX(),2) + pow(histogram.MeanY(),2) + C1) * (histogram.VarianceX()+histogram.VarianceY()+C2) );

        double DSSIM = (1-SSIM)/2;

        cout << z+1                             << ",";                   // patch number
        cout << sum                             << ",";                   // PSNR
        cout << SSIM                            << ",";                   // SSIM
        cout << DSSIM                           << ",";                   // DSSIM
        cout << histogram.MeanX()               << ",";                   // Mean of Patch
        cout << histogram.MeanY()               << ",";                   // Mean of Recon
        cout << histogram.VarianceX()           << ",";                   // Variance of patch
        cout << histogram.VarianceY()           << ",";                   // Variance of Recon
        cout << histogram.Covariance()          << ",";                   // Covariance
        cout << histogram.JointEntropy()        << ",";                   // JointEntropy (JE)
        cout << histogram.CrossCorrelation()    << ",";                   // Crosscorrelation (CC)
        cout << histogram.CorrelationRatioXY()  << ",";                   // Correlation Ratio (CR_X|Y: patch|recon)
        cout << histogram.CorrelationRatioYX()  << ",";                   // Correlation Ratio (CR_Y|X: recon|patch)
        cout << histogram.MutualInformation()   << ",";                   // Mutual Information (MI)
        cout << histogram.NormalizedMutualInformation() << ",";           // Normalized Mutual Information (NMI)
        cout << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << ",";  // Sums of squared diff. (SSD)

        if (nbins_x == nbins_y) {
            cout << histogram.LabelConsistency() << ",";                  // Label consistency (LC)
            cout << histogram.Kappa() << ",";                             // Kappa statistic (KS)
        }

        cout << endl;

    }  // end of patches loop


        // restore cout's original streambuf and close stream files
        cout.rdbuf(strm_buffer);


    // end of stacks loop
    cout << "done!" << endl;

}

template <typename T>
void irtkPatchBasedReconstruction<T>::Evaluate2d(int iter, string evaluationMaskName)
{

    printf("Iteration number %i - ", iter);;

    size_t found = evaluationMaskName.find("brain_1");
    string name_mask = "brain_1";
    if (found == -1) {found = evaluationMaskName.find("brain_2");   name_mask = "brain_2";};
    if (found == -1) {found = evaluationMaskName.find("placenta");  name_mask = "placenta";};
    if (found == -1) {found = evaluationMaskName.find("uterus");    name_mask = "uterus";};
    if (found == -1) {found = 0;};

    printf("Evaluate reconstruction for %.*s \n", 8, name_mask.c_str());

    // read the evaluation mask from disk
    irtkGenericImage<char> evaluationMask;
    evaluationMask.Read((char*)(evaluationMaskName.c_str()));

    // variables initialization
    char buffer[256];
    double widthx, widthy;
    // irtkImageAttributes mask_attr = m_mask.GetImageAttributes();
    Matrix4<float> maskW2I =  toMatrix4<float>(evaluationMask.GetWorldToImageMatrix());
    // char *histo_name  = 'evaluate_histogram';
    int nbins_x = 0, nbins_y = 0;   // Default number of bins for histogram

    // copy to host the GPU reconstructed volume -----------------------------------------------------------------------------------------------------
    m_GPURecon.checkGPUMemory();
    m_GPURecon.copyToHost(m_reconstruction->GetPointerToVoxels());
    irtkGenericImage<T> reconimage(m_reconstruction->GetImageAttributes());
    checkCudaErrors(cudaMemcpy(reconimage.GetPointerToVoxels(), m_GPURecon.getDataPtr(), reconimage.GetX()*reconimage.GetY()*reconimage.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));

    // sprintf(buffer, "evaluateReconimage.nii.gz");
    // reconimage.Write(buffer);

    // sprintf(buffer, "evaluateMaskimage.nii.gz");
    // evaluationMask.Write(buffer);

    // loop m_pbVolumes[i] -- stacks of patches  ------------------------------------------------------------------------------------------------------
    for (int i = 0; i <m_pbVolumes.size(); i++)
    {
        cout << "Evaluating stack no.[" << i+1 << "] "<< "------------------------------------------" << endl;

        // stream buffer memory
        streambuf* strm_buffer = cout.rdbuf();
        sprintf(buffer, "log-evaluate-stack-%i-iteration-%i-size-%i-%i-%.*s.csv", i, iter, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str()); // construct output stream files
        ofstream file(buffer);
        cout.rdbuf(file.rdbuf()); // assign streambuf to cout

        // prepare the evaluation file
        cout << "Stack[" << i << "]" << "//Patch no."   << ",";
        cout << "PSNR"                                  << ",";
        cout << "SSIM"                                  << ",";
        cout << "DSSIM"                                 << ",";
        cout << "PatchMean"                             << ",";
        cout << "ReconMean"                             << ",";
        cout << "PatchVariance"                         << ",";
        cout << "ReconVariance"                         << ",";
        cout << "Covariance"                            << ",";
        cout << "JointEntropy"                          << ",";
        cout << "Crosscorrelation"                      << ",";
        cout << "CorrelationRatioPatchRecon"            << ",";
        cout << "CorrelationRatioReconPatch"            << ",";
        cout << "MutualInformation"                     << ",";
        cout << "NormalizedMutualInformation"           << ",";
        cout << "SumSquareDiff"                         << ",";
        cout << "LabelConsistency"                      << ",";
        cout << "KappaStatistic"                        << ",";
        cout << endl;

        // update pb_volumes with the generated patches from the GPU
        m_pbVolumes[i].updateHostImagePatch2D();
        std::vector<ImagePatch2D<T> >     patchesTrans  = m_pbVolumes[i].getHostImagePatch2DVector();
        std::vector<irtkGenericImage<T> > patchesData   = m_pbVolumes[i].getHostImagePatchDataVector();

        // read weights
        unsigned int N = m_pbVolumes[i].getXYZPatchGridSize().x* m_pbVolumes[i].getXYZPatchGridSize().y* m_pbVolumes[i].getXYZPatchGridSize().z;
        irtkGenericImage<T> imgD(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
        checkCudaErrors(cudaMemcpy(imgD.GetPointerToVoxels(), m_pbVolumes[i].getSimWeightsPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));
        // std::vector<irtkGenericImage<T> > patchesWeight = m_pbVolumes[i].getHostImagePatchWeightVector();
        // patch.patchWeight

        // Set min and max of histogram ----------------------------------------------------------------------------------------------------------------------
        // Calculate number of bins to use
        if (nbins_x == 0) {
            nbins_x = (int) round(m_max_intensity - m_min_intensity) + 1;
            if (nbins_x > DEFAULT_BINS)
                nbins_x = DEFAULT_BINS;
        }
        if (nbins_y == 0) {
            nbins_y = (int) round(m_max_intensity - m_min_intensity) + 1;
            if (nbins_y > DEFAULT_BINS)
                nbins_y = DEFAULT_BINS;
        }

        // SSIM default settings
        // two variables to stabilize the division with weak denominator;
        // L the dynamic range of the pixel-values (typically this is 2^{\#bits_per_pixel}-1);
        // scriptstyle k_1 = 0.01 and  k_2 = 0.03 by default
        // for 255 color range C1 = 6.5025, C2 = 58.5225;
        double C1 = 6.5025, C2 = 58.5225;

        for (unsigned int z = 0; z < patchesData.size(); z++)
        {
            // Create histogram ----------------------------------------------------------------------------------------------------------------------------------------
            irtkHistogram_2D<int> histogram(nbins_x, nbins_y);
            widthx = (m_max_intensity - m_min_intensity) / (nbins_x - 1.0);
            widthy = (m_max_intensity - m_min_intensity) / (nbins_y - 1.0);

            histogram.PutMin(m_min_intensity - 0.5*widthx, m_min_intensity - 0.5*widthy);
            histogram.PutMax(m_max_intensity + 0.5*widthx, m_max_intensity + 0.5*widthy);


            ImagePatch2D<T>     patchTrans = patchesTrans[z];
            irtkGenericImage<T> patchData  = patchesData[z];
            // printPatch(patchData);
            double  sum       = 0;
            uint    numVoxels = 0;

            if (patchTrans.patchWeight<0.99999) continue;

            // printf("patchWeight = %f \n", patchTrans.patchWeight);
            // Fill histogram by looping all points in the patch --------------------------------------------------------------------------------------------------
            for (unsigned int y = 0; y < patchData.GetY(); y++) {
                for (unsigned int x = 0; x < patchData.GetX(); x++) {

                    // skip if weight is low
                    if (imgD(x,y,z)<0.99999) continue;

                    // skip if patch value is < 0
                    float3 patchPos   = make_float3(x, y, 0);
                    T patchVal = patchData(patchPos.x , patchPos.y , patchPos.z) ;
                    // printf("patchWeight = %f \t", patchTrans.patchWeight);
                    // printf("weight = %f \t", imgD(x,y,z));
                    // printf("patchVal = %f \n", patchVal);

                    if (patchVal<=0) continue; //skip if there is no values in the patch image at this location

                    // reconstruction space position
                    float3 _reconPos = m_GPURecon.reconstructedW2I* (patchTrans.Transformation * (patchTrans.I2W *  patchPos));
                    uint3  reconPos  = make_uint3(round_(_reconPos.x), round_(_reconPos.y), round_(_reconPos.z));
                    // mask space position
                    float3 _maskPos  = maskW2I * (patchTrans.Transformation * (patchTrans.I2W * patchPos));
                    uint3  maskPos   = make_uint3(round_(_maskPos.x), round_(_maskPos.y), round_(_maskPos.z));

                    // check if out of mask and FoV
                    if ((maskPos.x >= 0) && (maskPos.x < evaluationMask.GetX()) && (maskPos.y >= 0) && (maskPos.y < evaluationMask.GetY()) && (maskPos.z >= 0) && (maskPos.z < evaluationMask.GetZ())) {
                        if ((reconPos.x >= 0) && (reconPos.x < reconimage.GetX()) && (reconPos.y >= 0) && (reconPos.y < reconimage.GetY()) && (reconPos.z >= 0) && (reconPos.z < reconimage.GetZ())) {

                            char maskVal = evaluationMask(maskPos.x, maskPos.y, maskPos.z);
                            if (maskVal<=0) continue;   //skip if there is no values in the mask image at this location

                            T reconVal = reconimage(reconPos.x, reconPos.y, reconPos.z);
                            if (reconVal<=0) continue;   //skip if there is no values in the recon image at this location

                            histogram.AddSample(patchVal, reconVal);
                            sum += (patchVal - reconVal)*(patchVal - reconVal);
                            numVoxels +=1;

                        }
                    }

                }
            } // end of voxels loop in one patch

            // check if there are voxels in the patch, if not assign zeros for all evaluation metrics
            if (numVoxels<2){ continue; }


            sum = sum/numVoxels;
            sum = 20*log10(m_max_intensity) - 10*log10(sum);

            double SSIM = ((2*histogram.MeanX()*histogram.MeanY()+C1)*(2*histogram.Covariance()+C2))
                            / ( (pow(histogram.MeanX(),2) + pow(histogram.MeanY(),2) + C1) * (histogram.VarianceX()+histogram.VarianceY()+C2) );

            double DSSIM = (1-SSIM)/2;

            cout << z+1                             << ",";                   // patch number
            cout << sum                             << ",";                   // PSNR
            cout << SSIM                            << ",";                   // SSIM
            cout << DSSIM                           << ",";                   // DSSIM
            cout << histogram.MeanX()               << ",";                   // Mean of Patch
            cout << histogram.MeanY()               << ",";                   // Mean of Recon
            cout << histogram.VarianceX()           << ",";                   // Variance of patch
            cout << histogram.VarianceY()           << ",";                   // Variance of Recon
            cout << histogram.Covariance()          << ",";                   // Covariance
            cout << histogram.JointEntropy()        << ",";                   // JointEntropy (JE)
            cout << histogram.CrossCorrelation()    << ",";                   // Crosscorrelation (CC)
            cout << histogram.CorrelationRatioXY()  << ",";                   // Correlation Ratio (CR_X|Y: patch|recon)
            cout << histogram.CorrelationRatioYX()  << ",";                   // Correlation Ratio (CR_Y|X: recon|patch)
            cout << histogram.MutualInformation()   << ",";                   // Mutual Information (MI)
            cout << histogram.NormalizedMutualInformation() << ",";           // Normalized Mutual Information (NMI)
            cout << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << ",";  // Sums of squared diff. (SSD)

            if (nbins_x == nbins_y) {
                cout << histogram.LabelConsistency() << ",";                  // Label consistency (LC)
                cout << histogram.Kappa() << ",";                             // Kappa statistic (KS)
            }

            cout << endl;

        }  // end of patches loop


        // restore cout's original streambuf and close stream files
        cout.rdbuf(strm_buffer);

    }
    // end of stacks loop
    cout << "Evaluation done!" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;

}

template <typename T>
void irtkPatchBasedReconstruction<T>::EvaluateBaseline3d(string evaluationMaskName)
{

    char buffer[256];

    size_t found = evaluationMaskName.find("brain_1");
    string name_mask = "brain_1";
    if (found == -1) {found = evaluationMaskName.find("brain_2");   name_mask = "brain_2";};
    if (found == -1) {found = evaluationMaskName.find("placenta");  name_mask = "placenta";};
    if (found == -1) {found = evaluationMaskName.find("uterus");    name_mask = "uterus";};
    if (found == -1) {found = 0;};

    printf("Evaluate baseline for %.*s ", 8, name_mask.c_str());

    // define reference and target stacks
    int refIdx = 0; // reference stack
    int tarIdx = m_stacks.size()-1;

    irtkGenericImage<T> refStack    = m_stacks[refIdx];
    irtkGenericImage<T> tarStack    = m_stacks[tarIdx];
    // PatchBasedVolume<T> ref_pbVolume    = m_pbVolumes[refIdx];
    // PatchBasedVolume<T> tar_pbVolume    = m_pbVolumes[tarIdx];

    // read the evaluation mask from disk ----------------------
    irtkGenericImage<T> evaluationMask;
    evaluationMask.Read((char*)(evaluationMaskName.c_str()));

    // dilate the evaluation mask
    irtkDilation<T> dilation;
    dilation.SetConnectivity(CONNECTIVITY_26);
    dilation.SetInput(&evaluationMask);
    dilation.SetOutput(&evaluationMask);
    for (int i = 0; i < 3; i++) dilation.Run();


    // resample and crop around mask to avoid missing pixels
    irtkResampling<T> resampling(m_isoVoxelSize, m_isoVoxelSize, m_isoVoxelSize);
    irtkBSplineInterpolateImageFunction interpolatorGrey;
    irtkNearestNeighborInterpolateImageFunction interpolatorBinary;

    // resample and crop mask
    irtkGenericImage<char> m = evaluationMask;
    irtkRigidTransformation tmpTrans;
    TransformMask(evaluationMask, m, tmpTrans);
    CropImage(evaluationMask, m);
    resampling.SetInput(&evaluationMask);
    resampling.SetOutput(&evaluationMask);
    resampling.SetInterpolator(&interpolatorBinary);
    resampling.Run();
    sprintf(buffer, "evalMaskBaseline-%.*s.nii.gz", 8, name_mask.c_str());
    evaluationMask.Write(buffer);


    // resample and crop reference
    TransformMask(refStack, m, tmpTrans);
    CropImage(refStack, m);
    resampling.SetInput(&refStack);
    resampling.SetOutput(&refStack);
    resampling.SetInterpolator(&interpolatorGrey);
    resampling.Run();
    sprintf(buffer, "evalRefStackBaseline-%.*s.nii.gz", 8, name_mask.c_str());
    refStack.Write(buffer);

    m = evaluationMask;

    // resample and crop target
    TransformMask(tarStack, m, tmpTrans);
    CropImage(tarStack, m);
    resampling.SetInput(&tarStack);
    resampling.SetOutput(&tarStack);
    resampling.SetInterpolator(&interpolatorGrey);
    resampling.Run();
    sprintf(buffer, "evalTarStackBaseline-%.*s.nii.gz", 8, name_mask.c_str());
    tarStack.Write(buffer);

    m = evaluationMask;

    // dssim image
    irtkGenericImage<T> dssimImage  = m_stacks[refIdx];
    TransformMask(dssimImage, m, tmpTrans);
    CropImage(dssimImage, m);
    resampling.SetInput(&dssimImage);
    resampling.SetOutput(&dssimImage);
    resampling.SetInterpolator(&interpolatorGrey);
    resampling.Run();

    for (int z = dssimImage.GetZ() - 1; z >= 0; z--) {
        for (int y = dssimImage.GetY() - 1; y >= 0; y--) {
            for (int x = dssimImage.GetX() - 1; x >= 0; x--) {
                dssimImage(x, y, z) = 0.;
            }
        }
    }

    cout << "using stacks " <<  refIdx << " and " <<  tarIdx << " ... ";
    printf("evalMask size = %i %i %i - ", evaluationMask.GetX(),evaluationMask.GetY(),evaluationMask.GetZ());
    printf("dssimImg size = %i %i %i - ", dssimImage.GetX(),dssimImage.GetY(),dssimImage.GetZ());
    printf("refStack size = %i %i %i - ", refStack.GetX(),refStack.GetY(),refStack.GetZ());
    printf("tarStack size = %i %i %i \n", tarStack.GetX(),tarStack.GetY(),tarStack.GetZ());


    // Set min and max of histogram ----------------------------------------------------------------------------------------------------------------------
    // Calculate number of bins to use
    double  widthx, widthy;
    int nbins_x = 0, nbins_y = 0;   // Default number of bins for histogram
    if (nbins_x == 0) {
        nbins_x = (int) round(m_max_intensity - m_min_intensity) + 1;
        if (nbins_x > DEFAULT_BINS)
            nbins_x = DEFAULT_BINS;
    }
    if (nbins_y == 0) {
        nbins_y = (int) round(m_max_intensity - m_min_intensity) + 1;
        if (nbins_y > DEFAULT_BINS)
            nbins_y = DEFAULT_BINS;
    }

    irtkHistogram_2D<int> histogram(nbins_x, nbins_y);
    widthx = (m_max_intensity - m_min_intensity) / (nbins_x - 1.0);
    widthy = (m_max_intensity - m_min_intensity) / (nbins_y - 1.0);

    histogram.PutMin(m_min_intensity - 0.5*widthx, m_min_intensity - 0.5*widthy);
    histogram.PutMax(m_max_intensity + 0.5*widthx, m_max_intensity + 0.5*widthy);

    double  mse         = 0;
    uint    numVoxels   = 0;

    // SSIM default settings
    // two variables to stabilize the division with weak denominator;
    // L the dynamic range of the pixel-values (typically this is 2^{\#bits_per_pixel}-1);
    // scriptstyle k_1 = 0.01 and  k_2 = 0.03 by default
    // for 255 color range C1 = 6.5025, C2 = 58.5225;
    double C1 = 6.5025, C2 = 58.5225;
    double SSIM=0, DSSIM=0;

    printf("Silce # ");

    for (int z = evaluationMask.GetZ()-1; z >= 0; z--) {
        printf("%i ", z);
        for (int y = evaluationMask.GetY()-1; y >= 0; y--) {
            for (int x = evaluationMask.GetX()-1; x >= 0; x--) {
                if (((uint)evaluationMask(x,y,z))<=0) continue;

                // change to mask coordinates
                double xRef = x;
                double yRef = y;
                double zRef = z;
                evaluationMask.ImageToWorld(xRef, yRef, zRef);
                // refTransform.Transform(xRef, yRef, zRef);
                refStack.WorldToImage(xRef, yRef, zRef);
                // xRef = round(xRef);
                // yRef = round(yRef);
                // zRef = round(zRef);
                //if the voxel is inside mask ROI include it
                if ((xRef < 0) || (xRef >= refStack.GetX()) || (yRef < 0) || (yRef >= refStack.GetY()) || (zRef < 0) || (zRef >= refStack.GetZ())) continue;
                if (refStack(xRef,yRef,zRef)<=0) continue;

                //change to target coordinates
                double xTar = x;
                double yTar = y;
                double zTar = z;
                evaluationMask.ImageToWorld(xTar, yTar, zTar);
                // refTransform.Transform(xTar, yTar, zTar);
                tarStack.WorldToImage(xTar, yTar, zTar);
                // xTar = round(xTar);
                // yTar = round(yTar);
                // zTar = round(zTar);
                //if the voxel is inside mask ROI include it
                if ((xTar < 0) || (xTar >= tarStack.GetX()) || (yTar < 0) || (yTar >= tarStack.GetY()) || (zTar < 0) || (zTar >= tarStack.GetZ())) continue;
                if (tarStack(xTar,yTar,zTar)<=0) continue;
                // calculations
                T refValue = refStack(xRef,yRef,zRef);
                T tarValue = tarStack(xTar,yTar,zTar);

                histogram.AddSample(refValue, tarValue);

                // sum of the differences for PSNR calculation
                mse         += (refValue-tarValue)*(refValue-tarValue);
                numVoxels   += 1;

                // calculate ssim with window size 9x9
                double mu1=0, mu2=0, var1=0, var2=0, covar=0, num=0;
                double x_sq=0, y_sq=0, xy=0;
                // calculate means
                for (int zz = z+3; zz>=(z-3); zz--) {
                    int shiftz = zz-z;              // shift in z
                    int zz0 = zRef+shiftz;          // reference location
                    int zz1 = zTar+shiftz;          // target location
                    if ((zz<0) ||(zz >=evaluationMask.GetZ())) continue;
                    if ((zz0<0)||(zz0>=refStack.GetZ())) continue;
                    if ((zz1<0)||(zz1>=tarStack.GetZ())) continue;

                    for (int yy = y+3; yy>(y-3); yy--) {
                        int shifty = yy-y;          // shift in y
                        int yy0 = yRef+shifty;      // reference location
                        int yy1 = yTar+shifty;      // target location
                        if ((yy<0) ||(yy >=evaluationMask.GetY())) continue;
                        if ((yy0<0)||(yy0>=refStack.GetY())) continue;
                        if ((yy1<0)||(yy1>=tarStack.GetY())) continue;

                        for (int xx = x+3; xx>(x-3); xx--) {
                            int shiftx = xx-x;      // shift in x
                            int xx0 = xRef+shiftx;  // reference location
                            int xx1 = xTar+shiftx;  // target location
                            if ((xx<0) ||(xx >=evaluationMask.GetX())) continue;
                            if ((xx0<0)||(xx0>=refStack.GetX())) continue;
                            if ((xx1<0)||(xx1>=tarStack.GetX())) continue;

                            if ((evaluationMask(xx,yy,zz))<=0) continue;
                            if (refStack(xx0,yy0,zz0)<=0) continue;
                            if (tarStack(xx1,yy1,zz1)<=0) continue;

                            // printf("xx %i - yy %i - zz %i \n",xx,yy,zz);
                            mu1  += refStack(xx0,yy0,zz0);
                            mu2  += tarStack(xx1,yy1,zz1);
                            num  += 1;

                            x_sq += pow(refStack(xx0, yy0, zz0),2.0);
                            y_sq += pow(tarStack(xx1, yy1, zz1),2.0);
                            xy   += refStack(xx0, yy0, zz0)*tarStack(xx1, yy1, zz1);

                        }
                    }
                }
                mu1     = mu1/num;
                mu2     = mu2/num;
                var1    = (x_sq/num)- pow(mu1,2.0);
                var2    = (y_sq/num)- pow(mu2,2.0);
                covar   = (xy/num)  - mu1*mu2;

                double curSSIM = ((2*mu1*mu2+C1)*(2*covar+C2)) / ( (pow(mu1,2.)+pow(mu2,2.)+C1) * (var1+var2+C2) );
                SSIM    += curSSIM;
                DSSIM   += (1-curSSIM)/2;

                dssimImage(xRef, yRef, zRef) = (1-curSSIM)/2;
            }
        }
    }
    printf("..Done! \n");
    // save dssim image
    sprintf(buffer, "dssim-baseline-stacks-%i-%i-%.*s.nii.gz",refIdx, tarIdx, 8, name_mask.c_str()); // construct output stream files
    dssimImage.Write(buffer);

    SSIM    = SSIM/numVoxels;
    DSSIM   = DSSIM/numVoxels;

    // calculate PSNR
    mse     = mse/numVoxels;
    double psnr = 20*log10(m_max_intensity) - 10*log10(mse);

    printf("psnr = %f - mse = %f - SSIM = %f - DSSIM = %f ",psnr,mse,SSIM,DSSIM);
    printf("CC = %f - NMI = %f \n",histogram.CrossCorrelation(),histogram.NormalizedMutualInformation() );

    // open evaluation log file -----------------------------------------------------------------------------------------------------------
    // stream buffer memory
    streambuf* strm_buffer = cout.rdbuf();
    // sprintf(buffer, "log-evaluate-baseline-stacks-%i-%i-size-%i-%i-%.*s.csv", refIdx, tarIdx, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str()); // construct output stream files
	sprintf(buffer, "log-evaluate-%.*s.csv", 8, name_mask.c_str()); // construct output stream files
	ofstream file(buffer);
    cout.rdbuf(file.rdbuf()); // assign streambuf to cout

    // prepare the evaluation file
    cout << "patch-size-" << m_patchSize.x << "-stride-" << m_patchStride.x << ",";
    cout << "MSE"                                   << ",";
    cout << "PSNR"                                  << ",";
    cout << "SSIM"                                  << ",";
    cout << "DSSIM"                                 << ",";
    cout << "PatchMean"                             << ",";
    cout << "ReconMean"                             << ",";
    cout << "PatchVariance"                         << ",";
    cout << "ReconVariance"                         << ",";
    cout << "Covariance"                            << ",";
    cout << "JointEntropy"                          << ",";
    cout << "Crosscorrelation"                      << ",";
    cout << "CorrelationRatioPatchRecon"            << ",";
    cout << "CorrelationRatioReconPatch"            << ",";
    cout << "MutualInformation"                     << ",";
    cout << "NormalizedMutualInformation"           << ",";
    cout << "SumSquareDiff"                         << ",";
    cout << "LabelConsistency"                      << ",";
    cout << "KappaStatistic"                        << ",";
    cout << endl;

    cout << "baseline-stacks-" << refIdx << "-" << tarIdx << ",";
    cout << mse                             << ",";                   // MSE
    cout << psnr                            << ",";                   // PSNR
    cout << SSIM                            << ",";                   // SSIM
    cout << DSSIM                           << ",";                   // DSSIM
    cout << histogram.MeanX()               << ",";                   // Mean of Patch
    cout << histogram.MeanY()               << ",";                   // Mean of Recon
    cout << histogram.VarianceX()           << ",";                   // Variance of patch
    cout << histogram.VarianceY()           << ",";                   // Variance of Recon
    cout << histogram.Covariance()          << ",";                   // Covariance
    cout << histogram.JointEntropy()        << ",";                   // JointEntropy (JE)
    cout << histogram.CrossCorrelation()    << ",";                   // Crosscorrelation (CC)
    cout << histogram.CorrelationRatioXY()  << ",";                   // Correlation Ratio (CR_X|Y: patch|recon)
    cout << histogram.CorrelationRatioYX()  << ",";                   // Correlation Ratio (CR_Y|X: recon|patch)
    cout << histogram.MutualInformation()   << ",";                   // Mutual Information (MI)
    cout << histogram.NormalizedMutualInformation() << ",";           // Normalized Mutual Information (NMI)
    cout << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << ",";  // Sums of squared diff. (SSD)

    if (nbins_x == nbins_y) {
        cout << histogram.LabelConsistency() << ",";                  // Label consistency (LC)
        cout << histogram.Kappa() << ",";                             // Kappa statistic (KS)
    }

    cout << endl;

    // restore cout's original streambuf and close stream files
    cout.rdbuf(strm_buffer);

    cout << "Evaluation done!" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;

}

template <typename T>
void irtkPatchBasedReconstruction<T>::Evaluate3d(int iter, irtkGenericImage<T> reconimage, string evaluationMaskName)
{

    printf("Iteration number %i - ", iter);

    size_t found = evaluationMaskName.find("brain_1");
    string name_mask = "brain_1";
    if (found == -1) {found = evaluationMaskName.find("brain_2");   name_mask = "brain_2";};
    if (found == -1) {found = evaluationMaskName.find("placenta");  name_mask = "placenta";};
    if (found == -1) {found = evaluationMaskName.find("uterus");    name_mask = "uterus";};
    if (found == -1) {found = 0;};

    printf("Evaluate reconstruction for %.*s \n", 8, name_mask.c_str());

    // read the evaluation mask from disk ----------------------
    char buffer[256];
    irtkGenericImage<T> evaluationMask;
    evaluationMask.Read((char*)(evaluationMaskName.c_str()));

    // dilate the evaluation mask
    irtkDilation<T> dilation;
    dilation.SetConnectivity(CONNECTIVITY_26);
    dilation.SetInput(&evaluationMask);
    dilation.SetOutput(&evaluationMask);
    for (int i = 0; i < 3; i++) dilation.Run();


    // resample and crop around mask to avoid missing pixels
    irtkResampling<T> resampling(m_isoVoxelSize, m_isoVoxelSize, m_isoVoxelSize);
    irtkBSplineInterpolateImageFunction interpolatorGrey;
    irtkNearestNeighborInterpolateImageFunction interpolatorBinary;

    // resample and crop mask
    irtkGenericImage<char> charEvaluationMask = evaluationMask;
    irtkRigidTransformation tmpTrans;
    TransformMask(evaluationMask, charEvaluationMask, tmpTrans);
    CropImage(evaluationMask, charEvaluationMask);
    resampling.SetInput(&evaluationMask);
    resampling.SetOutput(&evaluationMask);
    resampling.SetInterpolator(&interpolatorBinary);
    resampling.Run();
    // sprintf(buffer, "evalMask-%.*s.nii.gz", 8, name_mask.c_str());
    // evaluationMask.Write(buffer);
    charEvaluationMask = evaluationMask;

    // variables initialization
    double widthx, widthy;
    Matrix4<float> maskW2I =  toMatrix4<float>(evaluationMask.GetWorldToImageMatrix());
    int nbins_x = 0, nbins_y = 0;   // initial number of bins for histogram

    // // update the recent reconstructed volume from GPU
    // m_GPURecon.checkGPUMemory();
    // m_GPURecon.copyToHost(m_reconstruction->GetPointerToVoxels());
    // irtkGenericImage<T> reconimage(m_reconstruction->GetImageAttributes());
    // checkCudaErrors(cudaMemcpy(reconimage.GetPointerToVoxels(), m_GPURecon.getDataPtr(), reconimage.GetX()*reconimage.GetY()*reconimage.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));

    // // save the recon image for debugging
    TransformMask(reconimage, charEvaluationMask, tmpTrans);
    CropImage(reconimage,charEvaluationMask);
    sprintf(buffer, "evalRecon-iter-%i-size-%i-stride%i-%.*s.nii.gz", iter, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str());
    reconimage.Write(buffer);

    // loop m_pbVolumes[i] -- stacks of patches  ------------------------------------------------------------------------------------------------------
    for (int i = 0; i <m_pbVolumes.size(); i++)
    {
        cout << "Evaluating stack no.[" << i+1 << "] "<< "------------------------------------------" << endl;

        // reconstruct using PSF
        ReconVolume<T> tmpGPURecon;

        m_GPURecon.copyToHost(m_reconstruction->GetPointerToVoxels());
        irtkGenericImage<char> reconmask = charEvaluationMask;
        irtkRigidTransformation transformation;
        TransformMask(*m_reconstruction, reconmask, transformation);

        tmpGPURecon.init(m_cuda_device, make_uint3(m_reconstruction->GetX(), m_reconstruction->GetY(), m_reconstruction->GetZ()), make_float3(m_reconstruction->GetXSize(), m_reconstruction->GetYSize(), m_reconstruction->GetZSize()), toMatrix4<float>(m_reconstruction->GetWorldToImageMatrix()), toMatrix4<float>(m_reconstruction->GetImageToWorldMatrix()));
        tmpGPURecon.setMask(reconmask.GetPointerToVoxels());
        tmpGPURecon.copyFromHost(m_reconstruction->GetPointerToVoxels());
        tmpGPURecon.reset();


        patchBasedPSFReconstruction_gpu(m_cuda_device, m_pbVolumes[i], tmpGPURecon,m_superpixel);
        tmpGPURecon.equalize();
        patchBasedSimulatePatches_gpu(m_cuda_device, m_pbVolumes[i], tmpGPURecon);

        tmpGPURecon.checkGPUMemory();
        // irtkGenericImage<T>*    tmpReconstruction;
        // tmpGPURecon.copyToHost(tmpReconstruction->GetPointerToVoxels());
        irtkGenericImage<T> reconimageFromSingleStack(m_reconstruction->GetImageAttributes());
        checkCudaErrors(cudaMemcpy(reconimageFromSingleStack.GetPointerToVoxels(), tmpGPURecon.getDataPtr(), reconimageFromSingleStack.GetX()*reconimageFromSingleStack.GetY()*reconimageFromSingleStack.GetZ()*sizeof(T), cudaMemcpyDeviceToHost));
        tmpGPURecon.reset();
        tmpGPURecon.release();
        // save the recon image for debugging
	    TransformMask(reconimageFromSingleStack, reconmask, tmpTrans);
	    CropImage(reconimageFromSingleStack,reconmask);
        sprintf(buffer, "evalStack%i-iter-%i-size-%i-stride%i-%.*s.nii.gz", i, iter, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str());
        reconimageFromSingleStack.Write(buffer);

        /* // Reconstruct recon image from single stack patches ----------------------------------------------------------------------------------------------
                // create a new target recon image for the current stack transformation
                irtkGenericImage<T> reconimageFromSingleStack = reconimage;

                for (int z = reconimageFromSingleStack.GetZ() - 1; z >= 0; z--) {
                    for (int y = reconimageFromSingleStack.GetY() - 1; y >= 0; y--) {
                        for (int x = reconimageFromSingleStack.GetX() - 1; x >= 0; x--) {
                            reconimageFromSingleStack(x, y, z) = 0.;
                        }
                    }
                }
                // create a counter image for calculating average redundant pixels
                irtkGenericImage<T> reconimageCounter = reconimageFromSingleStack;
                irtkGenericImage<T> patchWeightImage  = reconimageFromSingleStack;
                irtkGenericImage<T> voxelWeightImage  = reconimageFromSingleStack;

                // update pb_volumes with the generated patches from the GPU
                m_pbVolumes[i].updateHostImagePatch2D();
                std::vector<ImagePatch2D<T> >     patchesTrans  = m_pbVolumes[i].getHostImagePatch2DVector();
                std::vector<irtkGenericImage<T> > patchesData   = m_pbVolumes[i].getHostImagePatchDataVector();

                // read weights
                unsigned int N = m_pbVolumes[i].getXYZPatchGridSize().x* m_pbVolumes[i].getXYZPatchGridSize().y* m_pbVolumes[i].getXYZPatchGridSize().z;
                irtkGenericImage<T> voxelWeight(m_pbVolumes[i].getXYZPatchGridSize().x, m_pbVolumes[i].getXYZPatchGridSize().y, m_pbVolumes[i].getXYZPatchGridSize().z);
                checkCudaErrors(cudaMemcpy(voxelWeight.GetPointerToVoxels(), m_pbVolumes[i].getSimWeightsPtr(), N*sizeof(T), cudaMemcpyDeviceToHost));

                // reconstruct the 3D image from patches of stack[i]
                for (unsigned int z = 0; z < patchesData.size(); z++)
                {
                    ImagePatch2D<T>     patchTrans = patchesTrans[z];
                    irtkGenericImage<T> patchData  = patchesData[z];
                    // printf("patchTrans.patchWeight %i - %f \n", z, patchTrans.patchWeight);
                    if (patchTrans.patchWeight<0.99999) continue;   // skip patch if weight is low

                    for (unsigned int y = 0; y < patchData.GetY(); y++) {
                        for (unsigned int x = 0; x < patchData.GetX(); x++) {

                            // if (voxelWeight(x,y,z)<0.99999) continue;      // skip voxel if weight is low
                            if (patchData(x,y,0)<=0) continue;         // skip if there is no values in the patch image at this location

                            T patchVal       = patchData(x,y,0) ;
                            float3 patchPos  = make_float3(x,y,0);
                            float3 wpos      = patchTrans.Transformation * patchTrans.I2W * patchPos;//Tr*patch.I2W*patchPos;

                            // reconstruction space position
                            float3 _reconPos = m_GPURecon.reconstructedW2I * wpos;
                            uint3  reconPos  = make_uint3(round_(_reconPos.x), round_(_reconPos.y), round_(_reconPos.z));
                            // mask space position
                            // float3 _maskPos  = maskW2I * (patchTrans.Transformation * (patchTrans.I2W * patchPos));
                            float3 _maskPos  = maskW2I * wpos;
                            uint3  maskPos   = make_uint3(round_(_maskPos.x), round_(_maskPos.y), round_(_maskPos.z));

                            // check if out of mask and FoV
                            if ((maskPos.x >= 0) && (maskPos.x < evaluationMask.GetX()) && (maskPos.y >= 0) && (maskPos.y < evaluationMask.GetY()) && (maskPos.z >= 0) && (maskPos.z < evaluationMask.GetZ()))
                            {
                                if ((reconPos.x >= 0) && (reconPos.x < reconimage.GetX()) && (reconPos.y >= 0) && (reconPos.y < reconimage.GetY()) && (reconPos.z >= 0) && (reconPos.z < reconimage.GetZ()))
                                {
                                    char maskVal = evaluationMask(maskPos.x, maskPos.y, maskPos.z);
                                    if (maskVal<=0) continue;   //skip if there is no values in the mask image at this location

                                    reconimageFromSingleStack(reconPos.x,reconPos.y,reconPos.z)   += voxelWeight(x,y,z)*patchVal;    // add patch value to the new recon image
                                    reconimageCounter(reconPos.x,reconPos.y,reconPos.z)           += 1. ;         // increment counter
                                    voxelWeightImage(reconPos.x,reconPos.y,reconPos.z)            += voxelWeight(x,y,z);
                                    patchWeightImage(reconPos.x,reconPos.y,reconPos.z)            += patchTrans.patchWeight;
                                }
                            }
                        }
                    } // end of voxels loop in one patch
                } // end of patches loop in one stack

                // averaging redundant transfomed voxels
                for (int z = reconimageFromSingleStack.GetZ() - 1; z >= 0; z--) {
                    for (int y = reconimageFromSingleStack.GetY() - 1; y >= 0; y--) {
                        for (int x = reconimageFromSingleStack.GetX() - 1; x >= 0; x--) {
                            if (reconimageCounter(x, y, z)>1)
                            {
                                reconimageFromSingleStack(x, y, z)  = reconimageFromSingleStack(x, y, z)/ voxelWeightImage(x, y, z);
                                voxelWeightImage(x, y, z)           = voxelWeightImage(x, y, z)/ reconimageCounter(x, y, z);
                                patchWeightImage(x, y, z)           = patchWeightImage(x, y, z)/ reconimageCounter(x, y, z);
                            }
                        }
                    }
                }


                // save the initial recon image for debugging
                sprintf(buffer, "evaluateStack%i-%i.nii.gz", i,iter);
                reconimageFromSingleStack.Write(buffer);
                // save the initial counter recon image for debugging
                sprintf(buffer, "evaluateStack%i-%i-counter.nii.gz", i,iter);
                reconimageCounter.Write(buffer);
                // save the initial counter recon image for debugging
                sprintf(buffer, "evaluateStack%i-%i-voxelWeightImage.nii.gz", i,iter);
                voxelWeightImage.Write(buffer);
                // save the initial counter recon image for debugging
                sprintf(buffer, "evaluateStack%i-%i-patchWeightImage.nii.gz", i,iter);
                patchWeightImage.Write(buffer);
        */ // -----------------------------------------------------------------------------------------------------------------------------------------

        // Create histogram ----------------------------------------------------------------------------------------------------------------------------------------
        // Calculate number of bins to use
        if (nbins_x == 0) {
            nbins_x = (int) round(m_max_intensity - m_min_intensity) + 1;
            if (nbins_x > DEFAULT_BINS) nbins_x = DEFAULT_BINS;
        }
        if (nbins_y == 0) {
            nbins_y = (int) round(m_max_intensity - m_min_intensity) + 1;
            if (nbins_y > DEFAULT_BINS) nbins_y = DEFAULT_BINS;
        }

        irtkHistogram_2D<int> histogram(nbins_x, nbins_y);
        widthx = (m_max_intensity - m_min_intensity) / (nbins_x - 1.0);
        widthy = (m_max_intensity - m_min_intensity) / (nbins_y - 1.0);

        histogram.PutMin(m_min_intensity - 0.5*widthx, m_min_intensity - 0.5*widthy);
        histogram.PutMax(m_max_intensity + 0.5*widthx, m_max_intensity + 0.5*widthy);

        double  mse         = 0;
        uint    numVoxels   = 0;

        // SSIM default settings
        // two variables to stabilize the division with weak denominator;
        // L the dynamic range of the pixel-values (typically this is 2^{\#bits_per_pixel}-1);
        // scriptstyle k_1 = 0.01 and  k_2 = 0.03 by default
        // for 255 color range C1 = 6.5025, C2 = 58.5225;
        double C1 = 6.5025, C2 = 58.5225;
        double SSIM=0, DSSIM=0;

        // create a counter image for calculating average redundant pixels
        irtkGenericImage<T> dssimImage = reconimageFromSingleStack;
        // initialize the reconimageCounter with zeros
        for (int z = dssimImage.GetZ() - 1; z >= 0; z--) {
            for (int y = dssimImage.GetY() - 1; y >= 0; y--) {
                for (int x = dssimImage.GetX() - 1; x >= 0; x--) {
                    dssimImage(x, y, z) = 0.;
                }
            }
        }

        printf("reconimageFromSingleStack size = %i %i %i \n", reconimageFromSingleStack.GetX(),reconimageFromSingleStack.GetY(),reconimageFromSingleStack.GetZ());
        printf("reconimage size = %i %i %i \n", reconimage.GetX(),reconimage.GetY(),reconimage.GetZ());

        printf("Silce # ");
        for (int z = reconimageFromSingleStack.GetZ() - 1; z >= 0; z--) {
            printf("%i ", z);
            for (int y = reconimageFromSingleStack.GetY() - 1; y >= 0; y--) {
                for (int x = reconimageFromSingleStack.GetX() - 1; x >= 0; x--) {

                    if (reconimageFromSingleStack(x, y, z)<=0) continue;   //skip if there is no values in the mask image at this location
                    if (reconimage(x,y,z)<=0) continue;   //skip if there is no values in the recon image at this location

                    T stackValue = reconimageFromSingleStack(x,y,z);
                    T reconValue = reconimage(x,y,z);

                    histogram.AddSample(stackValue, reconValue);

                    // sum of the differences for PSNR calculation
                    mse         += (stackValue-reconValue)*(stackValue-reconValue);
                    numVoxels   +=1;

                    // calculate ssim with window size 9x9
                    double mu1=0, mu2=0, var1=0, var2=0, covar=0, num=0;
                    double x_sq=0, y_sq=0, xy=0;

                    // calculate means
                    // for (int zz = z+3; zz>z-3; zz--) {
                    //     if ((zz<0)||(zz>=reconimage.GetZ()))  continue;

                        for (int yy = y+3; yy>y-3; yy--) {
                            if ((yy<0)||(yy>=reconimage.GetY()))  continue;

                            for (int xx = x+3; xx>x-3; xx--) {
                                // skip if outside the image size
                                if ((xx<0)||(xx>=reconimage.GetX()))  continue;

                                if (reconimageFromSingleStack(xx, yy, z)<=0) continue;
                                if (reconimage(xx,yy,z)<=0) continue
                                	;

                                // printf("xx %i - yy %i - z %i \n",xx,yy,z);
                                mu1 += reconimageFromSingleStack(xx, yy, z);
                                mu2 += reconimage(xx,yy,z);
                                num += 1;

                                x_sq += pow(reconimageFromSingleStack(xx, yy, z),2.0);
                                y_sq += pow(reconimage(xx, yy, z),2.0);
                                xy   += reconimageFromSingleStack(xx, yy, z)*reconimage(xx, yy, z);

                            }
                        }
                    // }

                    mu1 	= mu1/num;
                    mu2 	= mu2/num;
                    var1    = (x_sq/num)- pow(mu1,2.0);
                    var2    = (y_sq/num)- pow(mu2,2.0);
                    covar   = (xy/num)  - mu1*mu2;

                    double curSSIM = ((2*mu1*mu2+C1)*(2*covar+C2)) / ( (pow(mu1,2.)+pow(mu2,2.)+C1) * (var1+var2+C2) );
                    SSIM    += curSSIM;
                    DSSIM   += (1-curSSIM)/2;

                    dssimImage(x, y, z) = (1-curSSIM)/2;
                }
            }
        }
        printf("..Done! \n");
        // save dssim image
        sprintf(buffer, "dssim-stack-%i-iter-%i-size-%i-stride%i-%.*s.nii.gz", i, iter, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str()); // construct output stream files
        dssimImage.Write(buffer);

        SSIM    = SSIM/numVoxels;
        DSSIM   = DSSIM/numVoxels;

        // calculate PSNR
        mse     = mse/numVoxels;
        double psnr = 20*log10(m_max_intensity) - 10*log10(mse);

        printf("psnr = %f - mse = %f - SSIM = %f - DSSIM = %f ",psnr,mse,SSIM,DSSIM);
        printf("CC = %f - NMI = %f \n",histogram.CrossCorrelation(),histogram.NormalizedMutualInformation() );

        // open evaluation log file -----------------------------------------------------------------------------------------------------------
        // stream buffer memory
        streambuf* strm_buffer = cout.rdbuf();
        // sprintf(buffer, "log-evaluate-stack-%i-iteration-%i-size-%i-%i-%.*s.csv", i, iter, m_patchSize.x, m_patchStride.x, 8, name_mask.c_str()); // construct output stream files
        sprintf(buffer, "log-evaluate-%.*s.csv", 8, name_mask.c_str()); // construct output stream files
        ofstream file(buffer, std::ios_base::app);
        cout.rdbuf(file.rdbuf()); // assign streambuf to cout

        // // prepare the evaluation file
        // cout << "Stack[" << i << "]" << "//Patch no."   << ",";
        // cout << "MSE"                                   << ",";
        // cout << "PSNR"                                  << ",";
        // cout << "SSIM"                                  << ",";
        // cout << "DSSIM"                                 << ",";
        // cout << "PatchMean"                             << ",";
        // cout << "ReconMean"                             << ",";
        // cout << "PatchVariance"                         << ",";
        // cout << "ReconVariance"                         << ",";
        // cout << "Covariance"                            << ",";
        // cout << "JointEntropy"                          << ",";
        // cout << "Crosscorrelation"                      << ",";
        // cout << "CorrelationRatioPatchRecon"            << ",";
        // cout << "CorrelationRatioReconPatch"            << ",";
        // cout << "MutualInformation"                     << ",";
        // cout << "NormalizedMutualInformation"           << ",";
        // cout << "SumSquareDiff"                         << ",";
        // cout << "LabelConsistency"                      << ",";
        // cout << "KappaStatistic"                        << ",";
        // cout << endl;

		cout << "iter-" << iter << "-stack-" << i << "-size-" << m_patchSize.x << "-stride-" << m_patchStride.x <<  ",";
        cout << mse                             << ",";                   // MSE
        cout << psnr                            << ",";                   // PSNR
        cout << SSIM                            << ",";                   // SSIM
        cout << DSSIM                           << ",";                   // DSSIM
        cout << histogram.MeanX()               << ",";                   // Mean of Patch
        cout << histogram.MeanY()               << ",";                   // Mean of Recon
        cout << histogram.VarianceX()           << ",";                   // Variance of patch
        cout << histogram.VarianceY()           << ",";                   // Variance of Recon
        cout << histogram.Covariance()          << ",";                   // Covariance
        cout << histogram.JointEntropy()        << ",";                   // JointEntropy (JE)
        cout << histogram.CrossCorrelation()    << ",";                   // Crosscorrelation (CC)
        cout << histogram.CorrelationRatioXY()  << ",";                   // Correlation Ratio (CR_X|Y: patch|recon)
        cout << histogram.CorrelationRatioYX()  << ",";                   // Correlation Ratio (CR_Y|X: recon|patch)
        cout << histogram.MutualInformation()   << ",";                   // Mutual Information (MI)
        cout << histogram.NormalizedMutualInformation() << ",";           // Normalized Mutual Information (NMI)
        cout << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << ",";  // Sums of squared diff. (SSD)

        if (nbins_x == nbins_y) {
            cout << histogram.LabelConsistency() << ",";                  // Label consistency (LC)
            cout << histogram.Kappa() << ",";                             // Kappa statistic (KS)
        }

        cout << endl;

        // restore cout's original streambuf and close stream files
        cout.rdbuf(strm_buffer);

    }
    // end of stacks loop
    cout << "Evaluation done!" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;

}


template <typename T>
void irtkPatchBasedReconstruction<T>::EvaluateGt3d(int iter, irtkGenericImage<T> reconimage)
{
	// evaluate reconstruction to ground truth in 3d

    char buffer[256];

    printf("Evaluate the reconstruction to ground truth image\n");

    // define reference and target stacks
	irtkGenericImage<T> tarStack = reconimage;
    irtkGenericImage<T> refStack;
    refStack.Read((char*)(m_evaluationGtName.c_str()));


    // dssim image
    irtkGenericImage<T> dssimImage  = refStack;

    for (int z = dssimImage.GetZ() - 1; z >= 0; z--) {
        for (int y = dssimImage.GetY() - 1; y >= 0; y--) {
            for (int x = dssimImage.GetX() - 1; x >= 0; x--) {
                dssimImage(x, y, z) = 0.;
            }
        }
    }

    printf("refStack size = %i %i %i - ", refStack.GetX(),refStack.GetY(),refStack.GetZ());
    printf("tarStack size = %i %i %i \n", tarStack.GetX(),tarStack.GetY(),tarStack.GetZ());

    // Set min and max of histogram ----------------------------------------------------------------------------------------------------------------------
    // Calculate number of bins to use
    double  widthx, widthy;
    int nbins_x = 0, nbins_y = 0;   // Default number of bins for histogram
    if (nbins_x == 0) {
        nbins_x = (int) round(m_max_intensity - m_min_intensity) + 1;
        if (nbins_x > DEFAULT_BINS)
            nbins_x = DEFAULT_BINS;
    }
    if (nbins_y == 0) {
        nbins_y = (int) round(m_max_intensity - m_min_intensity) + 1;
        if (nbins_y > DEFAULT_BINS)
            nbins_y = DEFAULT_BINS;
    }

    irtkHistogram_2D<int> histogram(nbins_x, nbins_y);
    widthx = (m_max_intensity - m_min_intensity) / (nbins_x - 1.0);
    widthy = (m_max_intensity - m_min_intensity) / (nbins_y - 1.0);

    histogram.PutMin(m_min_intensity - 0.5*widthx, m_min_intensity - 0.5*widthy);
    histogram.PutMax(m_max_intensity + 0.5*widthx, m_max_intensity + 0.5*widthy);

    double  mse         = 0;
    uint    numVoxels   = 0;

    // SSIM default settings
    // two variables to stabilize the division with weak denominator;
    // L the dynamic range of the pixel-values (typically this is 2^{\#bits_per_pixel}-1);
    // scriptstyle k_1 = 0.01 and  k_2 = 0.03 by default
    // for 255 color range C1 = 6.5025, C2 = 58.5225;
    double C1 = 6.5025, C2 = 58.5225;
    double SSIM=0, DSSIM=0;

    printf("Silce # ");

    for (int z = refStack.GetZ()-1; z >= 0; z--) {
        printf("%i ", z);
        for (int y = refStack.GetY()-1; y >= 0; y--) {
            for (int x = refStack.GetX()-1; x >= 0; x--) {

                if (((uint)refStack(x,y,z))<=0) continue;

                // change to mask coordinates
                double xRef = x;
                double yRef = y;
                double zRef = z;

                //change to target coordinates
                double xTar = x;
                double yTar = y;
                double zTar = z;
                refStack.ImageToWorld(xTar, yTar, zTar);
                tarStack.WorldToImage(xTar, yTar, zTar);
                //if the voxel is inside mask ROI include it
                if ((xTar < 0) || (xTar >= tarStack.GetX()) || (yTar < 0) || (yTar >= tarStack.GetY()) || (zTar < 0) || (zTar >= tarStack.GetZ())) continue;
                if (tarStack(xTar,yTar,zTar)<=0) continue;
                // calculations
                T refValue = refStack(xRef,yRef,zRef);
                T tarValue = tarStack(xTar,yTar,zTar);

                histogram.AddSample(refValue, tarValue);

                // sum of the differences for PSNR calculation
                mse         += (refValue-tarValue)*(refValue-tarValue);
                numVoxels   += 1;

                // calculate ssim with window size 9x9
                double mu1=0, mu2=0, var1=0, var2=0, covar=0, num=0;
                double x_sq=0, y_sq=0, xy=0;
                // calculate means
                for (int zz = z+3; zz>=(z-3); zz--) {
                    int shiftz = zz-z;              // shift in z
                    int zz0 = zRef+shiftz;          // reference location
                    int zz1 = zTar+shiftz;          // target location
                    if ((zz0<0)||(zz0>=refStack.GetZ())) continue;
                    if ((zz1<0)||(zz1>=tarStack.GetZ())) continue;

                    for (int yy = y+3; yy>(y-3); yy--) {
                        int shifty = yy-y;          // shift in y
                        int yy0 = yRef+shifty;      // reference location
                        int yy1 = yTar+shifty;      // target location
                        if ((yy0<0)||(yy0>=refStack.GetY())) continue;
                        if ((yy1<0)||(yy1>=tarStack.GetY())) continue;

                        for (int xx = x+3; xx>(x-3); xx--) {
                            int shiftx = xx-x;      // shift in x
                            int xx0 = xRef+shiftx;  // reference location
                            int xx1 = xTar+shiftx;  // target location
                            if ((xx0<0)||(xx0>=refStack.GetX())) continue;
                            if ((xx1<0)||(xx1>=tarStack.GetX())) continue;

                            if (refStack(xx0,yy0,zz0)<=0) continue;
                            if (tarStack(xx1,yy1,zz1)<=0) continue;

                            // printf("xx %i - yy %i - zz %i \n",xx,yy,zz);
                            mu1  += refStack(xx0,yy0,zz0);
                            mu2  += tarStack(xx1,yy1,zz1);
                            num  += 1;

                            x_sq += pow(refStack(xx0, yy0, zz0),2.0);
                            y_sq += pow(tarStack(xx1, yy1, zz1),2.0);
                            xy   += refStack(xx0, yy0, zz0)*tarStack(xx1, yy1, zz1);

                        }
                    }
                }

                mu1     = mu1/num;
                mu2     = mu2/num;
                var1    = (x_sq/num)- pow(mu1,2.0);
                var2    = (y_sq/num)- pow(mu2,2.0);
                covar   = (xy/num)  - mu1*mu2;

                double curSSIM = ((2*mu1*mu2+C1)*(2*covar+C2)) / ( (pow(mu1,2.)+pow(mu2,2.)+C1) * (var1+var2+C2) );
                SSIM    += curSSIM;
                DSSIM   += (1-curSSIM)/2;

                dssimImage(xRef, yRef, zRef) = (1-curSSIM)/2;
            }
        }
    }
    printf("..Done! \n");
    // save dssim image
    sprintf(buffer, "dssim-iter-%i-size-%i-%i.nii.gz", iter, m_patchSize.x, m_patchStride.x); // construct output stream files
    dssimImage.Write(buffer);

    SSIM    = SSIM/numVoxels;
    DSSIM   = DSSIM/numVoxels;

    // calculate PSNR
    mse     = mse/numVoxels;
    double psnr = 20*log10(m_max_intensity) - 10*log10(mse);

    printf("psnr = %f - mse = %f - SSIM = %f - DSSIM = %f ",psnr,mse,SSIM,DSSIM);
    printf("CC = %f - NMI = %f \n",histogram.CrossCorrelation(),histogram.NormalizedMutualInformation() );

    // open evaluation log file -----------------------------------------------------------------------------------------------------------
    // stream buffer memory
    streambuf* strm_buffer = cout.rdbuf();
    sprintf(buffer, "log-evaluate-Gt.csv"); // construct output stream files
    ofstream file(buffer, std::ios_base::app);
    cout.rdbuf(file.rdbuf()); // assign streambuf to cout

    // prepare the evaluation file
	if (iter==0)
	{
	    cout << "patch-size-" << m_patchSize.x << "-stride-" << m_patchStride.x << ",";
	    cout << "MSE"                                   << ",";
	    cout << "PSNR"                                  << ",";
	    cout << "SSIM"                                  << ",";
	    cout << "DSSIM"                                 << ",";
	    cout << "PatchMean"                             << ",";
	    cout << "ReconMean"                             << ",";
	    cout << "PatchVariance"                         << ",";
	    cout << "ReconVariance"                         << ",";
	    cout << "Covariance"                            << ",";
	    cout << "JointEntropy"                          << ",";
	    cout << "Crosscorrelation"                      << ",";
	    cout << "CorrelationRatioPatchRecon"            << ",";
	    cout << "CorrelationRatioReconPatch"            << ",";
	    cout << "MutualInformation"                     << ",";
	    cout << "NormalizedMutualInformation"           << ",";
	    cout << "SumSquareDiff"                         << ",";
	    cout << "LabelConsistency"                      << ",";
	    cout << "KappaStatistic"                        << ",";
	    cout << endl;
	}

    cout << "iter-" << iter << "-patch-size-" << m_patchSize.x << "-stride-" << m_patchStride.x << ",";
    cout << mse                             << ",";                   // MSE
    cout << psnr                            << ",";                   // PSNR
    cout << SSIM                            << ",";                   // SSIM
    cout << DSSIM                           << ",";                   // DSSIM
    cout << histogram.MeanX()               << ",";                   // Mean of Patch
    cout << histogram.MeanY()               << ",";                   // Mean of Recon
    cout << histogram.VarianceX()           << ",";                   // Variance of patch
    cout << histogram.VarianceY()           << ",";                   // Variance of Recon
    cout << histogram.Covariance()          << ",";                   // Covariance
    cout << histogram.JointEntropy()        << ",";                   // JointEntropy (JE)
    cout << histogram.CrossCorrelation()    << ",";                   // Crosscorrelation (CC)
    cout << histogram.CorrelationRatioXY()  << ",";                   // Correlation Ratio (CR_X|Y: patch|recon)
    cout << histogram.CorrelationRatioYX()  << ",";                   // Correlation Ratio (CR_Y|X: recon|patch)
    cout << histogram.MutualInformation()   << ",";                   // Mutual Information (MI)
    cout << histogram.NormalizedMutualInformation() << ",";           // Normalized Mutual Information (NMI)
    cout << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << ",";  // Sums of squared diff. (SSD)

    if (nbins_x == nbins_y) {
        cout << histogram.LabelConsistency() << ",";                  // Label consistency (LC)
        cout << histogram.Kappa() << ",";                             // Kappa statistic (KS)
    }

    cout << endl;

    // restore cout's original streambuf and close stream files
    cout.rdbuf(strm_buffer);

    cout << "Evaluation done!" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------" << endl;

}

template class irtkPatchBasedReconstruction < float > ;
template class irtkPatchBasedReconstruction < double > ;
