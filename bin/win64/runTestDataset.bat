set PATH=pwd

reconstruction_GPU2.exe -o 3TReconstruction.nii.gz -i ../../data/14_3T_nody_001.nii.gz ../../data/10_3T_nody_001.nii.gz ../../data/21_3T_nody_001.nii.gz ../../data/23_3T_nody_001.nii.gz -m ../../data/mask_10_3T_brain_smooth.nii.gz --disableBiasCorrection --useAutoTemplate --resolution 1.0 --useSINCPSF

