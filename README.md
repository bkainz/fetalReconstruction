# Fast motion compensation and super-resolution from multiple stacks of 2D slices

# PVR - Patch to Volume Reconstruction

**PVR** is a research-focused tool for the task of superresolution image reconstruction, developed at the [BioMedIA](https://biomedia.doc.ic.ac.uk/) research group. It provides command-line tools to reconstruct images from motion-corrupted imaging data.

# Dependencies
* [Boost](http://www.boost.org/) = v1.58.0
* [CUDA](https://developer.nvidia.com/cuda-toolkit) >= v 6.5.0
* [Intel TBB](https://www.threadingbuildingblocks.org/)

# Installation

```shell
$ cd <path-to-cudarecon/source> 
$ mkdir build
$ cd build
$ cmake ..
$ make 
```
# Usage

* PVR reconstruction using square patches
```shell
$ PVRreconstructionGPU -i <path-to-input-data> -o <reconstructed-image-filename> --resolution <1> --patchSize <x> <y> --patchStride <x> <y>
```

* PVR reconstruction using superpixels
```shell
$ PVRreconstructionGPU -i <path-to-input-data> -o <reconstructed-image-filename> --resolution <1> --superpixel --spxSize <x> --spxExtend <y>
```

* PVR reconstruction using square patches
```shell
$ SVRreconstructionGPU -i <path-to-input-data> -o <reconstructed-image-filename> --resolution <1> 
```

# References

* [[arxiv16]]() "PVR: Patch-to-Volume Reconstruction for Large Area Motion Correction of Fetal MRI." Amir Alansary, Bernhard Kainz, Martin Rajchl, Maria Murgasova,  Mellisa Damodaram, David F.A. Lloyd, Steven G. McDonagh, Alice Davidson, Mary Rutherford, Reza Razavi, Joseph V. Hajnal, and Daniel Rueckert.
* [[MICCAI15]](http://link.springer.com/chapter/10.1007/978-3-319-24571-3_66) "Flexible reconstruction and correction of unpredictable motion from stacks of 2D images." Bernhard Kainz, Amir Alansary, Christina Malamateniou, Kevin Keraudren, Mary Rutherford, Joseph V. Hajnal, and Daniel Rueckert.  In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 555-562. Springer International Publishing, 2015.
* [[MEDIA12]](http://www.sciencedirect.com/science/article/pii/S1361841512000965) "Reconstruction of fetal brain MRI with intensity matching and complete outlier removal." Maria Kuklisova-Murgasova, Gerardine Quaghebeur, Mary A. Rutherford, Joseph V. Hajnal, and Julia A. Schnabel. Medical image analysis 16, no. 8 (2012): 1550-1564.


# License

PVR is licensed under the [MIT license ![MIT](https://raw.githubusercontent.com/legacy-icons/license-icons/master/dist/32x32/mit.png)](http://opensource.org/licenses/MIT)

---------------------------------------------------


# SVR - Slice to Volume Reconstruction

**SVR** was used to produce the results shown in  

Bernhard Kainz, Markus Steinberger, Wolfgang Wein, Maria Kuklisova-Murgasova, 
Christina Malamateniou, Kevin Keraudren, Thomas Torsney-Weir, Mary Rutherford, 
Paul Aljabar, Joseph V. Hajnal, and Daniel Rueckert: Fast Volume Reconstruction 
from Motion Corrupted Stacks of 2D Slices. IEEE Transactions on Medical Imaging, 
in print, 2015. doi:10.1109/TMI.2015.2415453 


# Announcements

- The --useCPU flag has some bug currently and might have caused some confusion. CPU only does not  provide
   the full functionality of the GPU version. For example PSF sampling is different and the result of the 
   CPU version will be worse. -- I am working on it.
- Hacked a fix for the working CPU version  -- please report if you experience more bugs with that
- There seem to be an issue with CUDA 7.0. Please use CUDA 6.5 until I found the problem.
- The code requires refactoring to cleanly separate data from data access and CPU and GPU parts.


# Abstract

Moving objects cause motion artefacts when their enclosing volume is acquired as a stack of image slices. 
In this paper we present a fast multi-GPU accelerated implementation of slice-to-volume registration based super-resolution reconstruction with automatic outlier rejection and intensity bias correction. 
We introduce a novel fully automatic selection procedure for the image stack with the least motion, which will serve as an initial registration template. We fully evaluate our method and its high dimensional parameter space. Testing is done using artificially motion corrupted phantom datasets and using real world scenarios for the reconstruction of foetal organs from in-utero prenatal Magnetic Resonance Imaging and for the motion compensation of freehand compound Ultrasound data. 
On average we achieve a speed-up of more than 40x compared to a single CPU system, and another 1.70x for each additional GPU, while maintaining the same image quality as if calculated on a CPU. Our framework is qualitatively more accurate and on average $10\times$ times faster than currently available state-of-the-art multi-core methods.
The source code for this approach is open source and publicly available for download.

# Installation

The source code was successfully compiled on x86_64, CUDA 6.5, VS2012 (std-c++11), VS2013 (boost), and gcc 4.8. System: Intel Xeon E5-2630 v2 \@ 2.60GHz system with 16 GB RAM, an Nvidia Tesla K40 with 12 GB RAM and a Geforce 780 Graphics card with 6 GB RAM and also tested on a Nvidia Titan GPU.

## Necessary Hardware requirements
* x86_64 CPU

## Optional Hardware requirements
* Nvidia GPU compute capability >= 3.0 
In case of no GPU: only CPU reconstruction will be available. Use --useCPU flag [CUDA is still necessary to compile!]
HOWEVER: The CPU version only allows a simplified PSF evaluation (only linearly low sampled Gaussian PSF)! 
ONLY THE GPU ACCELERATED VERSION WILL PROVIDE ALL IN THE PAPER DESCRIBED FEATURES!

## Necessary third party libraries
* Nvidia CUDA >= 6.0 https://developer.nvidia.com/cuda-downloads
* Boost (compiled libraries, we used 1.55.0) http://www.boost.org/users/download/
* Intel's TBB https://www.threadingbuildingblocks.org/

## Optional third party libraries
* CULA http://www.culatools.com/
* IRTK http://www.doc.ic.ac.uk/~dr/software/ (This is already included in the source code repository as minimal build lib. We only use image handling and CPU registration functionality from IRTK)

# BUILD INSTRUCTIONS

* go to the source directory
* make a new directory "build"
* execute cmake .. in this directory (give paths to TBB and other third party libs if necessary using cmakegui .. or ccmake ..)
* make or open the Visual Studio Solution 

# Usage

* get two or more overlapping stacks of image slices
* [optional]: generate a region of interest and sace as .nii or .nii.gz
* run the program SVRreconstructionGPU (previously reconstruction_GPU2)
* Example: ./SVRreconstructionGPU -o 3T_GPUtest.nii -i ../../data/14_3T_nody_001.nii.gz ../../data/10_3T_nody_001.nii.gz ../../data/23_3T_nody_001.nii.gz ../../data/21_3T_nody_001.nii.gz -m ../../data/mask_10_3T_brain_smooth.nii.gz
* Options:

  -h [ --help ]                      Print usage messages

  -o [ --output ] arg                Name for the reconstructed volume. Nifti
                                     or Analyze format.

  -m [ --mask ] arg                  Binary mask to define the region od
                                     interest. Nifti or Analyze format.

  -i [ --input ] arg                 [stack_1] .. [stack_N]  The input stacks.
                                     Nifti or Analyze format.

  -t [ --transformation ] arg        The transformations of the input stack to
                                     template in 'dof' format used in IRTK.
                                     Only rough alignment with correct
                                     orienation and some overlap is needed. Use
                                     'id' for an identity transformation for at
                                     least one stack. The first stack with 'id'
                                     transformation  will be resampled as
                                     template.

  --thickness arg                    [th_1] .. [th_N] Give slice
                                     thickness.[Default: twice voxel size in z
                                     direction]

  -p [ --packages ] arg              Give number of packages used during
                                     acquisition for each stack. The stacks
                                     will be split into packages during
                                     registration iteration 1 and then into odd
                                     and even slices within each package during
                                     registration iteration 2. The method will
                                     then continue with slice to  volume
                                     approach. [Default: slice to volume
                                     registration only]

  --iterations arg (=4)              Number of registration-reconstruction
                                     iterations.

  --sigma arg (=12)                  Stdev for bias field. [Default: 12mm]
  --resolution arg (=0.75)           Isotropic resolution of the volume.
                                     [Default: 0.75mm]

  --multires arg (=3)                Multiresolution smooting with given number
                                     of levels. [Default: 3]

  --average arg (=700)               Average intensity value for stacks
                                     [Default: 700]

  --delta arg (=150)                  Parameter to define what is an edge.
                                     [Default: 150]

  --lambda arg (=0.02)                 Smoothing parameter. [Default: 0.02]

  --lastIterLambda arg (=0.01)       Smoothing parameter for last iteration.
                                     [Default: 0.01]

  --smooth_mask arg (=4)             Smooth the mask to reduce artefacts of
                                     manual segmentation. [Default: 4mm]

  --global_bias_correction arg (=0)  Correct the bias in reconstructed image
                                     against previous estimation.

  --low_intensity_cutoff arg (=0.01) Lower intensity threshold for inclusion of
                                     voxels in global bias correction.

  --force_exclude arg                force_exclude [number of slices] [ind1]
                                     ... [indN]  Force exclusion of slices with
                                     these indices.

  --no_intensity_matching arg        Switch off intensity matching.

  --log_prefix arg                   Prefix for the log file.

  --debug arg (=0)                    Debug mode - save intermediate results.

  --debug_gpu                         Debug only GPU results.

  --rec_iterations_first arg (=4)     Set number of superresolution iterations

  --rec_iterations_last arg (=13)     Set number of superresolution iterations
                                     for the last iteration

  --num_stacks_tuner arg (=0)          Set number of input stacks that are
                                     really used (for tuner evaluation, use
                                     only first x)

  --no_log arg (=0)                    Do not redirect cout and cerr to log
                                     files.

  -d [ --devices ] arg                 Select the CP > 3.0 GPUs on which the
                                     reconstruction should be executed.
                                     Default: all devices > CP 3.0

  --tfolder arg                      [folder] Use existing slice-to-volume
                                     transformations to initialize the
                                     reconstruction.

  --sfolder arg                      [folder] Use existing registered slices
                                     and replace loaded ones (have to be
                                     equally many as loaded from stacks).

  --referenceVolume arg              Name for an optional reference volume.
                                     Will be used as inital reconstruction.

  --T1PackageSize arg                is a test if you can register T1 to T2
                                     using NMI and only one iteration

  --useCPU                           use CPU for reconstruction and
                                     registration; performs superresolution and
                                     robust statistics on CPU. Default is using
                                     the GPU

  --useCPUReg                        use CPU for more flexible CPU
                                     registration; performs superresolution and
                                     robust statistics on GPU. [default, best
                                     result]

  --useGPUReg                        use faster but less accurate and flexible
                                     GPU registration; performs superresolution
                                     and robust statistics on GPU.

  --useAutoTemplate                  select 3D registration template stack
                                     automatically with matrix rank method.

  --disableBiasCorrection            disable bias field correction for cases
                                     with little or no bias field
                                     inhomogenities (makes it faster but less
                                     reliable for stron intensity bias)



# License

PVR is licensed under the [MIT license ![MIT](https://raw.githubusercontent.com/legacy-icons/license-icons/master/dist/32x32/mit.png)](http://opensource.org/licenses/MIT)
