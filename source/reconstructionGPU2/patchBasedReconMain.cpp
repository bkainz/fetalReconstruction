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
#include <iostream>
#include <boost/program_options.hpp>
#include <irtkPatchBasedReconstruction.h>

#include <irtkResampling.h>

namespace po = boost::program_options;

#define OFF 0
#define ON 1

//current test
//-i 14_3T_nody_001.nii.gz 21_3T_nody_001.nii.gz -o testPSFRecon1.nii -m mask_10_3T_brain_smooth_dilated.nii.gz --thickness 2.5 2.5
//-i 14_3T_nody_001.nii.gz 21_3T_nody_001.nii.gz -o testPSFRecon1.nii -m mask_10_3T_brain_smooth_dilated.nii.gz --debug
//-i 14_3T_nody_001.nii.gz 21_3T_nody_001.nii.gz -o testPSFRecon1.nii -m mask_10_3T_brain_smooth_dilated.nii.gz --iterations 1 --sr_iterations 1 --debug
//-i 14_3T_nody_001.nii.gz 21_3T_nody_001.nii.gz 10_3T_nody_001.nii.gz 23_3T_nody_001.nii.gz -o testPSFRecon1.nii -m mask_10_3T_brain_smooth_dilated.nii.gz --iterations 1 --sr_iterations 1 --debug
//-i 14_3T_nody_001_d_2.nii.gz 14_3T_nody_001_d_2.nii.gz -o testPSFRecon1.nii -m mask_10_3T_brain_smooth_dilated_d_2.nii.gz --iterations 1 --sr_iterations 1 --resolution 1 --debug
//-o 201507091115_PHILIPS484E467_2801_WIPchest_recon.nii.gz -i 201507091115_PHILIPS484E467_2801_WIPchest_tse_ms_oblSENSE.nii.gz 201507091115_PHILIPS484E467_901_WIPchest_tse_ms_sagSENSE.nii.gz 201507091115_PHILIPS484E467_1001_WIPchest_tse_ms_corSENSE.nii.gz 201507091115_PHILIPS484E467_1101_WIPchest_tse_ms_traSENSE.nii.gz 201507091115_PHILIPS484E467_1401_WIPchest_tse_ms_oblSENSE.nii.gz 201507091115_PHILIPS484E467_1601_WIPchest_tse_ms_oblSENSE.nii.gz 201507091115_PHILIPS484E467_2701_WIPchest_tse_ms_oblSENSE.nii.gz 201507091115_PHILIPS484E467_2901_WIPchest_tse_ms_oblSENSE.nii.gz 201507091115_PHILIPS484E467_3301_WIPchest_tse_ms_oblSENSE.nii.gz 201507091115_PHILIPS484E467_3801_WIPchest_tse_ms_oblSENSE.nii.gz -m 201507091115_PHILIPS484E467_901_WIPchest_tse_ms_sagSENSE_mask.nii.gz --resolution 0.4 --sr_iterations 9  --iterations 9
//D:\zwi\201507091115_PHILIPS484E467_aortaTest\test\chest\withObl
//TODO problem with custom thickness in registration
//-o iFIND2_201507031509_PHILIPS484E467_1103_DelRecT2_brain_recon.nii.gz -i iFIND2_201507031509_PHILIPS484E467_1103_DelRecT2_brain_sag.nii.gz iFIND2_201507031509_PHILIPS484E467_1303_DelRecT2_brain_tra.nii.gz iFIND2_201507031509_PHILIPS484E467_1403_DelRecT2_brain_cor.nii.gz -m iFIND2_201507031509_PHILIPS484E467_1103_DelRecT2_brain_sag_mask.nii.gz --resolution 0.6 --sr_iterations 5 --iterations 7 -p 4 4 4
//-i /home/bk14/cudarecon/data/14_3T_nody_001.nii.gz /home/bk14/cudarecon/data/21_3T_nody_001.nii.gz -o testPSFRecon1.nii -m /home/bk14/cudarecon/data/mask_10_3T_brain_smooth_dilated.nii.gz --iterations 1 --sr_iterations 1 --debug

//TODO add option to do registration (CPU) with ful slices (no patch splitting) -- will unify tools

int main(int argc, char** argv)
{

    //TODO IRTK data I/O implementation and parameters
    //TODO boost I/O
    //TODO GUI
    int cuda_device = 0;

    vector<string> inputStacks;
    vector<string> inputTransformations;
    string outputName;
    string maskName;
    string evaluationGtName;
    vector<string> evaluationMaskNames;
    int nStacks;
    vector<irtkGenericImage<float> > stacks;
    vector<irtkRigidTransformation> stack_transformations;

    //TODO for anatomical improvements, recompute patches per iteration and make patches getting larger per iteration
    vector<float> thickness;
    irtkGenericImage<char> mask;
    // vector < irtkGenericImage<char> > evaluationMask;
    std::vector<unsigned int> patchSize;
    std::vector<unsigned int> patchStride;
    unsigned int spxSize;
    unsigned int spxExtend;

    vector<int> devicesToUse; //TODO multi GPU currently only first in list
    vector<int> packages;
    string existingReconstructionTargetName;

    // general default configurations
    float output_resolution = 0.75;
    bool _debug 			= false;
    bool _patch_extraction	= false;
    int sr_iterations 		= 7;
    int iterations 			= 7;
    int _dilateMask       	= 0;
    bool useFullSlices 		= false;
    bool _noMatchIntensities= false;
    bool _superpixel        = false;
    bool _hierarchical      = false;
    bool _resample          = false;
    bool _evaluateBaseline  = false;

    // patch-based default configurations
    patchSize.resize(2);
    patchStride.resize(2);
    patchSize[0] 			= 32;
    patchSize[1] 			= 32;
    patchStride[0] 			= 16;
    patchStride[1] 			= 16;
    
    // superpixel default configurations
    spxSize 				= 16;
    spxExtend 				= 50; // 50% percent
    
    try
    {
        po::options_description desc("Options");
        desc.add_options()
            ("help,h", "Print usage messages")
            ("output,o", po::value<string>(&outputName)->required(), "Name for the reconstructed volume. Nifti or Analyze format.")
            ("mask,m", po::value<string>(&maskName), "Binary mask to define the region od interest. Nifti or Analyze format.")
            ("input,i", po::value<vector<string> >(&inputStacks)->multitoken(), "[stack_1] .. [stack_N]  The input stacks. Nifti or Analyze format.")
            ("existingReconTarget,e", po::value<string>(&existingReconstructionTargetName), "Set an existing reconstruction as target image.")
            ("patchSize", po::value< vector<unsigned int> >(&patchSize)->multitoken(), "defines the size of the 2D patches for patchBased reconstruction")
            ("patchStride", po::value< vector<unsigned int> >(&patchStride)->multitoken(), "defines the stride of the 2D patches for patchBased reconstruction")
            ("resolution", po::value< float >(&output_resolution)->default_value(0.75), "Isotropic output resolution of the volume. [Default: 0.75mm]")
            ("transformation,t", po::value< vector<string> >(&inputTransformations)->multitoken(), "The transformations of the input stack to template in \'dof\' format used in IRTK. Only rough alignment with correct orienation and some overlap is needed. Use \'id\' for an identity transformation for at least one stack. The first stack with \'id\' transformation  will be resampled as template.")
            ("noMatchIntensities", po::bool_switch(&_noMatchIntensities)->default_value(false), "Skip match intensities between the input stacks")
            ("superpixel,s", po::bool_switch(&_superpixel)->default_value(false), "Turn on superpixel-based reconstruction. [Default: false]")
            ("spxSize", po::value< unsigned int >(&spxSize)->multitoken(), "defines the initial size (<=64) of the 2D superpixels for the superpixel-based reconstruction. [Default: 16]")
            ("spxExtend", po::value< unsigned int >(&spxExtend)->multitoken(), "defines a ratio [0-100]%% from a superpixel size for dilation or overlapping. [Default: 50%%]")
            ("hierarchical", po::bool_switch(&_hierarchical)->default_value(false), "Turn on hierarchical-based reconstruction. [Default: false]")
            ("debug", po::bool_switch(&_debug)->default_value(false), "Write debug images.")
            ("resample", po::bool_switch(&_resample)->default_value(false), "Resample input stacks before reconstruction [Note: consumes larger memory].")
            ("useFullSlices", po::bool_switch(&useFullSlices)->default_value(false), "Use full slices instead of patches.")
            ("dilateMask", po::value< int >(&_dilateMask)->default_value(0), "Dilate reconstruction mask n-iterations.")
            ("iterations", po::value< int >(&iterations)->default_value(7), "number of registration iterations.")
            ("sr_iterations", po::value< int >(&sr_iterations)->default_value(7), "number of Super-resolution iterations.")
            ("packages,p", po::value< vector<int> >(&packages)->multitoken(), "Give number of packages used during acquisition for each stack. The stacks will be split into packages during registration iteration 1 and then into odd and even slices within each package during registration iteration 2. The method will then continue with slice to  volume approach. [Default: slice to volume registration only]")
            ("devices,d", po::value< vector<int> >(&devicesToUse)->multitoken(), "Select the CP > 3.0 GPUs on which the reconstruction should be executed. Default: all devices > CP 3.0")
            ("thickness", po::value< vector<float> >(&thickness)->multitoken(), "[th_1] .. [th_N] Give patch thickness.[Default: twice voxel size in z direction] -- TODO")
            ;

        if (EVALUATE){
            desc.add_options()
                ("evaluateGt", po::value<string>(&evaluationGtName)->multitoken(), "Ground truth filename for reconstruction evaluation.")
                ("evaluation,v", po::value<vector<string> >(&evaluationMaskNames)->multitoken(), "Binary mask to define the region of interest for evaluation. Nifti or Analyze format.")
                ("evaluateBaseline", po::bool_switch(&_evaluateBaseline)->default_value(false), "Evaluate baseline - measure the evaluation metrics between the first and last stack before the reconstruction.")
                ("patchExtraction", po::bool_switch(&_patch_extraction)->default_value(false), "Evaluate patch extraction only.")
                ;
            }
        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm); // can throw

            if (vm.count("help"))
            {
                std::cout << "Application to perform reconstruction of volumetric MRI from thick patches." << std::endl
                          << desc << std::endl;
                return EXIT_SUCCESS;
            }

            po::notify(vm);
        }
        catch (po::error& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Unhandled exception while parsing arguments:  "
                  << e.what() << ", application will now exit" << std::endl;
        return EXIT_FAILURE;
    }
    string testn = outputName;

    if (devicesToUse.size() == 0)
        devicesToUse.push_back(0);

    //TODO multi GPU
    cudaSetDevice(devicesToUse[0]);
    cuda_device = devicesToUse[0]; //select one device (should be fixed to use multi-GPU) 

    cout << "Reconstructed volume name ... " << outputName << endl;
    nStacks = inputStacks.size();
    cout << "Number of stacks ... " << nStacks << endl;

    bool setThickness = false;
    if (thickness.empty())
        setThickness = true;

    vector<int> packagesN;
    vector<float> thicknessN;

    for (int i = 0; i < nStacks; i++)
    {
        irtkGenericImage<float> stack;
        stack.Read(inputStacks[i].c_str());

#if 0
        //TODO test for registration improvement
        // resamples to recon space -- deactivate for faster recon but less accurate registration
        irtkImageAttributes attr = stack.GetImageAttributes();
        //TODO customized slice thickness
        irtkResamplingWithPadding<float> resampling(output_resolution, output_resolution, attr._dz, -1);
        resampling.SetInput(&stack);
        resampling.SetOutput(&stack);
        resampling.Run();
        //test end
#endif
        double dx, dy, dz;
        stack.GetPixelSize(&dx, &dy, &dz);
        if (setThickness)
        {
            thickness.push_back(dz);
        }
        else
        {
            thickness[i] = thickness[i] / 2.0; //given that it is called thickness and to maintain same scheme as earlier versions
        }

        //stack.PutPixelSize(dx, dy, thickness[i]);

        cout << "Reading stack ... " << inputStacks[i] << " thickness " << thickness[i] << " * 2 " << endl;

        //separate 4D volumes
        if (stack.GetT() > 1)
        {
            irtkImageAttributes attr = stack.GetImageAttributes();
            attr._t = 1;
            for (int t = 0; t < stack.GetT(); t++)
            {
                cout << "Splitting stack ... " << inputStacks[i] << endl;
                irtkGenericImage<float> stackn(attr);
                memcpy(stackn.GetPointerToVoxels(), stack.GetPointerToVoxels() + t*stack.GetX()*stack.GetY()*stack.GetZ(),
                       stack.GetX()*stack.GetY()*stack.GetZ() * sizeof(float));
                stacks.push_back(stackn);

                thicknessN.push_back(thickness[i]);
                if (!packages.empty())
                    packagesN.push_back(packages[i]);
            }
        }
        else
        {
            stacks.push_back(stack);
            if (!thickness.empty())
                thicknessN.push_back(thickness[i]);
            if (!packages.empty())
                packagesN.push_back(packages[i]);

            else
            {
                printf("using standard package size\n");
            }
        }
    }


    thickness = thicknessN;
    if (!packages.empty())
        packages = packagesN;

    nStacks = stacks.size();

    /*for (int i = 0; i < stacks.size(); i++)
        {
        char buffer[255];
        sprintf(buffer, "test%i.nii", i);
        stacks[i].Write(buffer);
        }
        std::cin.get();*/
    
    printf("useFullSlices %s \n", useFullSlices ? "true" : "false");

    if (_hierarchical)
    {
        if ( _superpixel && !useFullSlices ) { 
            cout << "hierarchical-superpixel-based ON" << endl; 
	        patchSize[0] 	= spxSize;
	        patchSize[1] 	= spxSize;
	        patchStride[0] 	= spxExtend;
            patchStride[1] 	= spxExtend;
        }else{ 
            if (useFullSlices) {
                printf("SVR ON - useFullSlices \n");
                _hierarchical   = false;
            }else{
                cout << "hierarchical-patch-based ON" << endl;

            }
        }
    }else{
        if ( _superpixel && !useFullSlices ) { 
            cout << "superpixel-based ON" << endl; 
            patchSize[0] 	= spxSize;
	        patchSize[1] 	= spxSize;
	        patchStride[0] 	= spxExtend;
            patchStride[1] 	= spxExtend;
        }else{ 
            if (useFullSlices) {
                printf("SVR ON - useFullSlices \n");
            }else{
                cout << "patch-based ON" << endl; 
            }
        }
    }

    cout << "cuda_dev = " << cuda_device << endl;


    // TODO: intenral implementation
    if (!_hierarchical)
    {   
        irtkPatchBasedReconstruction<float> reconstruction(cuda_device, output_resolution, make_uint2(patchSize[0], patchSize[1]), make_uint2(patchStride[0], patchStride[1]), 
        iterations, sr_iterations, _dilateMask, _resample, _noMatchIntensities, _superpixel, _hierarchical, _debug, _patch_extraction);   
		
        reconstruction.setImageStacks(stacks, thicknessN, inputTransformations, packagesN);
        reconstruction.setUseFullSlices(useFullSlices);

        if (packagesN.size() == nStacks)
        {
            reconstruction.setPackageDenominators(packagesN);
            printf("setting package denominators...\n");
        }

        if (!maskName.empty())
        {
            mask.Read((char*)(maskName.c_str()));
            reconstruction.setMask(mask);
        }

        reconstruction.setEvaluationBaseline(_evaluateBaseline);

        if (!evaluationMaskNames.empty())
        {   
            // read organ masks for evaluating the reconstruction quality at these regions
            reconstruction.setEvaluationMaskName(evaluationMaskNames);
        }

        if (!evaluationGtName.empty())
        {   
            // read organ masks for evaluating the reconstruction quality at these regions
            reconstruction.setEvaluationGtName(evaluationGtName);
        }


        if (!existingReconstructionTargetName.empty())
        {
            irtkGenericImage<float>* existingReconstructionTarget = new irtkGenericImage<float>();
            existingReconstructionTarget->Read(existingReconstructionTargetName.c_str());
            reconstruction.setExistingReconstructionTarget(existingReconstructionTarget);
        }

        reconstruction.run();

   		if (_patch_extraction) return 0; // evaluate patch extraction only

        reconstruction.getReconstruction()->Write(outputName.c_str());

    }else{
        // hierarchical-based

        for (int iter = 0; iter <= (iterations); iter++)
        {   

            printf("Hierarchical level - %i  -------------------------------- \n", iter); 

            int recon_iter = 1;
            if (_patch_extraction) recon_iter = iterations;


            irtkPatchBasedReconstruction<float> reconstruction_h(cuda_device, output_resolution, make_uint2(patchSize[0], patchSize[1]), make_uint2(patchStride[0], patchStride[1]), recon_iter, sr_iterations, _dilateMask, _resample, _noMatchIntensities, _superpixel, _hierarchical, _debug, _patch_extraction);   
            reconstruction_h.setImageStacks(stacks, thicknessN, inputTransformations, packagesN);
            reconstruction_h.setUseFullSlices(useFullSlices);

            if (packagesN.size() == nStacks)
            {
                reconstruction_h.setPackageDenominators(packagesN);
                printf("setting package denominators...\n");
            }

            if (!maskName.empty())
            {
                mask.Read((char*)(maskName.c_str()));
                reconstruction_h.setMask(mask);
            }

            if (!evaluationMaskNames.empty()) { reconstruction_h.setEvaluationMaskName(evaluationMaskNames); } // read organ masks for evaluating the reconstruction quality at these regions
            if (!evaluationGtName.empty()) { reconstruction_h.setEvaluationGtName(evaluationGtName); } // read organ masks for evaluating the reconstruction quality at these regions

            
            if (iter==0) 
            {
                if (!existingReconstructionTargetName.empty())
                {	
                	reconstruction_h.setEvaluationBaseline(_evaluateBaseline);
                    irtkGenericImage<float>* existingReconstructionTarget = new irtkGenericImage<float>();
                    existingReconstructionTarget->Read(existingReconstructionTargetName.c_str());
                    reconstruction_h.setExistingReconstructionTarget(existingReconstructionTarget);
                }
            }else{

                // update existing reconstruction target
                char newReconstructionTargetName[256];
                
                if ( !_superpixel) {
                	sprintf(newReconstructionTargetName, "reconimage1_%i_%i.nii.gz", patchSize[0]+4,patchStride[0]+2);
            	} else { 
            		sprintf(newReconstructionTargetName, "reconimage1_%i_%i.nii.gz", patchSize[0]+4,patchStride[0]);
            	}

                printf("Existing Reconstruction Target Name %s \n", newReconstructionTargetName);
                irtkGenericImage<float>* existingReconstructionTarget = new irtkGenericImage<float>();
                existingReconstructionTarget->Read(newReconstructionTargetName);
                reconstruction_h.setExistingReconstructionTarget(existingReconstructionTarget);
            }
    
            reconstruction_h.run();
            reconstruction_h.setEvaluationBaseline(false);
    	
			if (_patch_extraction) return 0;


            if (iter<iterations) 
            {
                printf("Reconstruction is done! ... increase patch size and repeat \n");
                // increase patch size
                patchSize[0]   -= 4; patchSize[1]   -= 4;
                if ( !_superpixel) { patchStride[0] -= 2; patchStride[1] -= 2; }
            }else{
                printf("Final reconstruction is done! ... save output image and exit \n");
                reconstruction_h.getReconstruction()->Write(outputName.c_str());
            }

            // reconstruction_h.release();
            
        }
    }

    return EXIT_SUCCESS;

}
