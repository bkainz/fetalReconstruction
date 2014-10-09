/*=========================================================================

Library   : Image Registration Toolkit (IRTK)
Module    : $Id: irtkReconstructionCuda.cc 1 2013-11-15 14:36:30 bkainz $
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
Version   : $Revision: 1 $
Changes   : $Author: bkainz $

Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
Maria Murgasova, Kevin Keraudren
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

=========================================================================*/


#include "irtkEvaluation.h"

#define DEFAULT_BINS 255

EvalResult irtkEvaluation::evaluate()
{
	EvalResult result;

	irtkTransformation *transformation = NULL;
	irtkInterpolateImageFunction *interpolator = NULL;
	irtkRealPixel target_min, source_min, target_max, source_max;
	int ok, i, x, y, z, i1, j1, k1, i2, j2, k2, verbose;
	double x1, y1, z1, x2, y2, z2, Tp, widthx, widthy, val;

	verbose = false;

	Tp = -1.0 * FLT_MAX;

	// Fix ROI
	i1 = 0;
	j1 = 0;
	k1 = 0;
	i2 = target.GetX();
	j2 = target.GetY();
	k2 = target.GetZ();

	// Fix no. of bins;
	nbins_x = 0;
	nbins_y = 0;

	interpolator = new irtkLinearInterpolateImageFunction;
	irtkGreyImage mask;
	mask = target;

	irtkGreyPixel *ptr2mask = mask.GetPointerToVoxels();
	irtkRealPixel *ptr2tgt  = target.GetPointerToVoxels();

	for (i = 0; i < target.GetNumberOfVoxels(); i++) {
		if (*ptr2tgt > Tp)
			*ptr2mask = 1;
		else
			*ptr2mask = 0;

		++ptr2tgt;
		++ptr2mask;
	}

	// If there is an region of interest, use it
	if ((i1 != 0) || (i2 != target.GetX()) ||
		(j1 != 0) || (j2 != target.GetY()) ||
		(k1 != 0) || (k2 != target.GetZ())) {
			target = target.GetRegion(i1, j1, k1, i2, j2, k2);
			source = source.GetRegion(i1, j1, k1, i2, j2, k2);
			mask   = mask.GetRegion(i1, j1, k1, i2, j2, k2);
	}

	// Set min and max of histogram
	target.GetMinMax(&target_min, &target_max);
	source.GetMinMax(&source_min, &source_max);
	if (verbose == true) {
		cout << "Min and max of X is " << target_min
			<< " and " << target_max << endl;
		cout << "Min and max of Y is " << source_min
			<< " and " << source_max << endl;
	}

	// Calculate number of bins to use
	if (nbins_x == 0) {
		nbins_x = (int) round(target_max - target_min) + 1;
		if (nbins_x > DEFAULT_BINS)
			nbins_x = DEFAULT_BINS;
	}

	if (nbins_y == 0) {
		nbins_y = (int) round(source_max - source_min) + 1;
		if (nbins_y > DEFAULT_BINS)
			nbins_y = DEFAULT_BINS;
	}

	// Create default interpolator if necessary
	if (interpolator == NULL) {
		if (source.GetZ() == 1){
			interpolator = new irtkNearestNeighborInterpolateImageFunction2D;
		} else {
			interpolator = new irtkNearestNeighborInterpolateImageFunction;
		}
	}
	interpolator->SetInput(&source);
	interpolator->Initialize();

	// Calculate the source image domain in which we can interpolate
	interpolator->Inside(x1, y1, z1, x2, y2, z2);

	// Create histogram
	irtkHistogram_2D<int> histogram(nbins_x, nbins_y);
	widthx = (target_max - target_min) / (nbins_x - 1.0);
	widthy = (source_max - source_min) / (nbins_y - 1.0);

	histogram.PutMin(target_min - 0.5*widthx, source_min - 0.5*widthy);
	histogram.PutMax(target_max + 0.5*widthx, source_max + 0.5*widthy);

	// 	if (trans_name == 0) {
	// 		transformation = new irtkRigidTransformation;
	// 	} else {
	transformation = new irtkRigidTransformation;
	//}

	target_min = FLT_MAX;
	source_min = FLT_MAX;
	target_max = -1.0 * FLT_MAX;
	source_max = -1.0 * FLT_MAX;

	double sum = 0;

	// Fill histogram
	for (z = 0; z < target.GetZ(); z++) {
		for (y = 0; y < target.GetY(); y++) {
			for (x = 0; x < target.GetX(); x++) {

				if (mask(x, y, z) > 0) {
					val = target(x, y, z);

					if (val > target_max)
						target_max = val;
					if (val < target_min)
						target_min = val;

					irtkPoint p(x, y, z);
					// Transform point into world coordinates
					target.ImageToWorld(p);
					// Transform point
					transformation->Transform(p);
					// Transform point into image coordinates
					source.WorldToImage(p);

					// A bad thing might happen for the 2D case.
					if ((source.GetZ() == 1) &&
						(p._z > 0.5 || p._z < -0.5)){
							cerr << "Transformed point outside plane of 2D source image." << endl;
							exit(1);
					}

					// 2D and in plane but out of FoV.
					if ((source.GetZ() == 1) &&
						(p._x <= x1 || p._x >= x2 ||
						p._y <= y1 || p._y >= y2))
						continue;

					// 3D and out of FoV.
					if ((source.GetZ() > 1) &&
						(p._x <= x1 || p._x >= x2 ||
						p._y <= y1 || p._y >= y2 ||
						p._z <= z1 || p._z >= z2))
						continue;

					// Should be able to interpolate if we've got this far.

					val = interpolator->EvaluateInside(p._x, p._y, p._z);

					histogram.AddSample(target(x, y, z), val);

					sum += (target(x, y, z) - val)*(target(x, y, z) - val);

					if (val >  source_max)
						source_max = val;
					if (val < source_min)
						source_min = val;

				}
			}
		}
		// 		if (histo_name != NULL) {
		// 			histogram.WriteAsImage(histo_name);
		// 		}
	}

	sum = sum/target.GetNumberOfVoxels();

	sum = 20*log10(target_max) - 10*log10(sum);

	if (verbose == true) {
		cout << "ROI Min and max of X is " << target_min << " and " << target_max << endl;
		cout << "ROI Min and max of Y is " << source_min << " and " << source_max << endl;
		cout << "Number of bins  X x Y : " << histogram.NumberOfBinsX() << " x " << histogram.NumberOfBinsY() << endl;
		cout << "Number of Samples: "     << histogram.NumberOfSamples() << endl;
		cout << "Mean of X: "             << histogram.MeanX() << endl;
		cout << "Mean of Y: "             << histogram.MeanY() << endl;
		cout << "Variance of X: "         << histogram.VarianceX() << endl;
		cout << "Variance of Y: "         << histogram.VarianceY() << endl;
		cout << "Covariance: "            << histogram.Covariance() << endl;
		cout << "Crosscorrelation: "      << histogram.CrossCorrelation() << endl;
		cout << "Sums of squared diff.: " << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << endl;
		cout << "Marginal entropy of X: " << histogram.EntropyX() << endl;
		cout << "Marginal entropy of Y: " << histogram.EntropyY() << endl;
		cout << "Joint entropy: "         << histogram.JointEntropy() << endl;
		cout << "Mutual Information: "    << histogram.MutualInformation() << endl;
		cout << "Norm. Mutual Information: " << histogram.NormalizedMutualInformation() << endl;
		cout << "Correlation ratio C(X|Y): " << histogram.CorrelationRatioXY() << endl;
		cout << "Correlation ratio C(Y|X): " << histogram.CorrelationRatioYX() << endl;
		if (nbins_x == nbins_y) {
			cout << "Label consistency: " << histogram.LabelConsistency() << endl;
			cout << "Kappa statistic: " << histogram.Kappa() << endl;
		}
		cout << "PSNR: "   << sum << endl;
	} else {
		cout << "CC: "     << histogram.CrossCorrelation() << endl;
		cout << "SSD: "    << histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples() << endl;
		cout << "JE: "     << histogram.JointEntropy() << endl;
		cout << "MI: "     << histogram.MutualInformation() << endl;
		cout << "NMI: "    << histogram.NormalizedMutualInformation() << endl;
		cout << "CR_X|Y: " << histogram.CorrelationRatioXY() << endl;
		cout << "CR_Y|X: " << histogram.CorrelationRatioYX() << endl;
		if (nbins_x == nbins_y) {
			cout << "LC: "   << histogram.LabelConsistency() << endl;
			cout << "KS: "   << histogram.Kappa() << endl;
		}
		cout << "PSNR: "   << sum << endl;
	}

	result.cc = histogram.CrossCorrelation();
	result.ssd = histogram.SumsOfSquaredDifferences() / (double)histogram.NumberOfSamples();
	result.je = histogram.JointEntropy();
	result.mi = histogram.MutualInformation();
	result.nmi = histogram.NormalizedMutualInformation();
	result.cr_x = histogram.CorrelationRatioXY();
	result.cr_y = histogram.CorrelationRatioYX();
	if (nbins_x == nbins_y) {
		result.lc = histogram.LabelConsistency();
		result.ks = histogram.Kappa();
	}
	result.psnr = sum;
	result.varX = histogram.VarianceX();
	result.meanX = histogram.MeanX();
	result.jentropy = histogram.JointEntropy();
	return result;
}