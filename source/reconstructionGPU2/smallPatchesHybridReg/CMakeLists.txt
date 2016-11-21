#TODO make new library

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cuda_add_library(patchBasedSlightlyAcceleratedIRTKRigidRegLib 
		irtkImageRigidRegistrationWithPadding_hybrid.cc
		irtkImageRigidRegistrationWithPadding_hybrid.h
		ccSimilarityMetric.cu
		ccSimilarityMetric.cuh
		)