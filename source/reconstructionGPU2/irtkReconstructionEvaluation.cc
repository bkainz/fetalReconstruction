/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id$
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date$
  Version   : $Revision$
  Changes   : $Author$

=========================================================================*/

#include <irtkImage.h>

#include <irtkImageFunction.h>

#include <irtkHistogram.h>

#include <irtkTransformation.h>

#include <irtkRegistration2.h>

// Default filenames
char *source_name = NULL, *target_name = NULL, *output_name = NULL;
char *diff_name = NULL, *mask_name = NULL, *dofout_name = NULL;
char *trans_name  = NULL, *histo_name  = NULL;

#define DEFAULT_BINS 255

// Default number of bins
int nbins_x = 0, nbins_y = 0;


void usage()
{
  cerr << "Usage: Evaluate [target] [source] [output] [diff_volume] [mask] [dofout] <options>\n" << endl;
  cerr << "where <options> is one or more of the following: \n" << endl;
  cerr << "<-histo file>      Output histogram" << endl;
  cerr << "<-nbins_x no>      Number of bins for target (Default smaller of dynamic range or 255)" << endl;
  cerr << "<-nbins_y no>      Number of bins for source (Default as above)" << endl;
  cerr << "<-Rx1 pixel>       Region of interest" << endl;
  cerr << "<-Ry1 pixel>       Region of interest" << endl;
  cerr << "<-Rz1 pixel>       Region of interest" << endl;
  cerr << "<-Rx2 pixel>       Region of interest" << endl;
  cerr << "<-Ry2 pixel>       Region of interest" << endl;
  cerr << "<-Rz2 pixel>       Region of interest" << endl;
  cerr << "<-Tp  value>       Padding value in target" << endl;
  cerr << "<-linear>          Linear interpolation" << endl;
  cerr << "<-bspline>         B-spline interpolation" << endl;
  cerr << "<-cspline>         Cubic spline interpolation" << endl;
  cerr << "<-sinc>            Sinc interpolation" << endl;
  exit(1);
}


void cropImage()
{
  irtkRealImage target(target_name);
  irtkRealImage source(source_name);
  irtkRealImage cropped_target(source_name);
  irtkRealImage diff_image(source_name);
  irtkGreyImage mask(source_name);

  int x, y, z;

  cout << "Crop Image ... "; cout.flush();

  // Crop target with source image and generate difference volume
  for (z = 0; z < source.GetZ(); z++){     
    for (y = 0; y < source.GetY(); y++){       
      for (x = 0; x < source.GetX(); x++){
        
        irtkPoint p(x, y, z);   // source index
        // Crop target and generate mask from the source image
        if (source(x,y,z) > 0) {
           mask(x,y,z) = 1;
           // Transform point into world coordinates
           source.ImageToWorld(p);
           // Transform point to target image coordinates
           target.WorldToImage(p);
           // Transform point into image coordinates
           if (p._x>0 && p._y>0 && p._z>0 && p._x<target.GetX() && p._y<target.GetY() && p._z<target.GetZ()){
             cropped_target(x,y,z) = target(p._x,p._y,p._z);
             diff_image(x,y,z)     = cropped_target(x,y,z)-source(x,y,z);
           }
        }else{
          diff_image(x,y,z) = -1;
          mask(x,y,z)       = 0;
        }
      }
    }
  }
  // Save cropped target image to disk
  cropped_target.Write(output_name);
  diff_image.Write(diff_name);
  mask.Write(mask_name);
}


void rreg2()
{
  irtkRealImage target(output_name);
  irtkRealImage source(source_name);
  irtkGreyImage mask(mask_name);

  int ok, padding;
  int target_x1, target_y1, target_z1, target_x2, target_y2, target_z2;
  int source_x1, source_y1, source_z1, source_x2, source_y2, source_z2;
  double tox, toy, toz, sox, soy, soz;
  tox = toy = toz = sox = soy = soz = 0.0;
  int centerImages = false, mask_dilation = 0;
  irtkMatrix tmat(4, 4);
  irtkMatrix smat(4, 4);
  irtkMatrix tempMat, transfMat;
  tmat.Ident();
  smat.Ident();

  cout << "Registration ... "; cout.flush();


  // Create transformation
  irtkTransformation *transformation = new irtkRigidTransformation;

  // Create registration
  irtkImageRegistration2 *registration = new irtkImageRigidRegistration2;

  // Fix ROI
  target_x1 = 0;
  target_y1 = 0;
  target_z1 = 0;
  target_x2 = target.GetX();
  target_y2 = target.GetY();
  target_z2 = target.GetZ();
  source_x1 = 0;
  source_y1 = 0;
  source_z1 = 0;
  source_x2 = source.GetX();
  source_y2 = source.GetY();
  source_z2 = source.GetZ();

  // Default parameters
  padding = MIN_GREY;


  if (mask.GetX() != target.GetX() || mask.GetY() != target.GetY() || mask.GetZ() != target.GetZ()) 
  {
      cerr << "Mask given does not match target dimensions." << endl;
      exit(1);
  }

  // Is there a mask to use?
  if (mask_name != NULL) {
    int voxels, i;
    irtkRealPixel *ptr2target, *ptr2mask;
    irtkRealImage mask(mask_name);

    if (mask.GetX() != target.GetX() ||
        mask.GetY() != target.GetY() ||
        mask.GetZ() != target.GetZ()) {
      cerr << "Mask given does not match target dimensions." << endl;
      exit(1);
    }

    if (mask_dilation > 0) {
      irtkDilation<irtkRealPixel> dilation;
      dilation.SetConnectivity(CONNECTIVITY_26);
      dilation.SetInput(&mask);
      dilation.SetOutput(&mask);
      cout << "Dilating mask ... ";
      cout.flush();
      for (i = 0; i < mask_dilation; i++) dilation.Run();
      cout << "done" << endl;

    }

    voxels     = target.GetNumberOfVoxels();
    ptr2target = target.GetPointerToVoxels();
    ptr2mask   = mask.GetPointerToVoxels();

    for (i = 0; i < voxels; ++i) {
      if (*ptr2mask <= 0) {
        *ptr2target = padding;
      }
      ++ptr2mask;
      ++ptr2target;
    }
  }

  // If there is a region of interest, use it
  if ((target_x1 != 0) || (target_x2 != target.GetX()) ||
      (target_y1 != 0) || (target_y2 != target.GetY()) ||
      (target_z1 != 0) || (target_z2 != target.GetZ())) {
    target = target.GetRegion(target_x1, target_y1, target_z1,
                              target_x2, target_y2, target_z2);
  }

  // If there is an region of interest for the source image, use it
  if ((source_x1 != 0) || (source_x2 != source.GetX()) ||
      (source_y1 != 0) || (source_y2 != source.GetY()) ||
      (source_z1 != 0) || (source_z2 != source.GetZ())) {
    source = source.GetRegion(source_x1, source_y1, source_z1,
                              source_x2, source_y2, source_z2);
  }

  if (centerImages == true) {
    cout << "Centering ... ";
    // Place the voxel centre at the world origin.
    target.GetOrigin(tox, toy, toz);
    source.GetOrigin(sox, soy, soz);
    target.PutOrigin(0.0, 0.0, 0.0);
    source.PutOrigin(0.0, 0.0, 0.0);

    // Update the transformation accordingly.
    tmat(0, 3) = tox;
    tmat(1, 3) = toy;
    tmat(2, 3) = toz;
    smat(0, 3) = -1.0 * sox;
    smat(1, 3) = -1.0 * soy;
    smat(2, 3) = -1.0 * soz;

    irtkRigidTransformation *rigidTransf = dynamic_cast<irtkRigidTransformation*> (transformation);

    transfMat = rigidTransf->GetMatrix();
    tempMat   = transfMat * tmat;
    tempMat   = smat * tempMat;

    rigidTransf->PutMatrix(tempMat);
    transformation = rigidTransf;
    cout << "done" << endl;
  }

  // Set input and output for the registration filter
  registration->SetInput(&target, &source);
  registration->SetOutput(transformation);

  // Make an initial Guess for the parameters.
  registration->GuessParameter();

  if (padding != MIN_GREY) {
    registration->SetTargetPadding(padding);
  }

  // Run registration filter
  registration->Run();

  // Write the final transformation estimate
  if (dofout_name != NULL) {
    if (centerImages == false) {
      transformation->Write(dofout_name);
    } else {
      // Undo the effect of centering the images.
      irtkRigidTransformation *rigidTransf = dynamic_cast<irtkRigidTransformation*> (transformation);

      tmat(0, 3) = -1.0 * tox;
      tmat(1, 3) = -1.0 * toy;
      tmat(2, 3) = -1.0 * toz;
      smat(0, 3) = sox;
      smat(1, 3) = soy;
      smat(2, 3) = soz;

      transfMat = rigidTransf->GetMatrix();
      tempMat   = transfMat * tmat;
      tempMat   = smat * tempMat;

      rigidTransf->PutMatrix(tempMat);
      rigidTransf->irtkTransformation::Write(dofout_name);
    }
  }
}


int main(int argc, char **argv)
{
  irtkTransformation *transformation = NULL;
  irtkInterpolateImageFunction *interpolator = NULL;
  irtkRealPixel target_min, source_min, target_max, source_max;
  int ok, i, x, y, z, i1, j1, k1, i2, j2, k2, verbose;
  double x1, y1, z1, x2, y2, z2, Tp, widthx, widthy, val;

  // Check command line
  if (argc < 3) {
    usage();
  }

  // Parse source and target images
  target_name = argv[1];
  argc--;
  argv++;
  source_name = argv[1];
  argc--;
  argv++;
  output_name = argv[1];
  argc--;
  argv++;
  diff_name = argv[1];
  argc--;
  argv++;
  mask_name = argv[1];
  argc--;
  argv++;
  dofout_name = argv[1];
  argc--;
  argv++;

  // Read target and source image

  // crop and register target and source images
  cropImage();
  rreg2();

  irtkRealImage target(output_name);
  irtkRealImage source(source_name);
  irtkGreyImage mask(mask_name);
  trans_name = dofout_name;

  // Default options
  verbose = false;

  // Default padding
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

  while (argc > 1) {
    ok = false;
    if ((ok == false) && (strcmp(argv[1], "-verbose") == 0)) {
      argc--;
      argv++;
      verbose = true;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-histo") == 0)) {
      argc--;
      argv++;
      histo_name = argv[1];
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-nbins_x") == 0)) {
      argc--;
      argv++;
      nbins_x = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-nbins_y") == 0)) {
      argc--;
      argv++;
      nbins_y = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Tp") == 0)) {
      argc--;
      argv++;
      Tp = atof(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Rx1") == 0)) {
      argc--;
      argv++;
      i1 = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Rx2") == 0)) {
      argc--;
      argv++;
      i2 = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Ry1") == 0)) {
      argc--;
      argv++;
      j1 = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Ry2") == 0)) {
      argc--;
      argv++;
      j2 = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Rz1") == 0)) {
      argc--;
      argv++;
      k1 = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-Rz2") == 0)) {
      argc--;
      argv++;
      k2 = atoi(argv[1]);
      argc--;
      argv++;
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-linear") == 0)) {
      argc--;
      argv++;
      if (source.GetZ() == 1){
        interpolator = new irtkLinearInterpolateImageFunction2D;
      } else {
        interpolator = new irtkLinearInterpolateImageFunction;
      }
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-bspline") == 0)) {
      argc--;
      argv++;
      if (source.GetZ() == 1){
      	interpolator = new irtkBSplineInterpolateImageFunction2D;
      } else {
      	interpolator = new irtkBSplineInterpolateImageFunction;
      }
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-cspline") == 0)) {
      argc--;
      argv++;
      if (source.GetZ() == 1){
      	interpolator = new irtkCSplineInterpolateImageFunction2D;
      } else {
      	interpolator = new irtkCSplineInterpolateImageFunction;
      }
      ok = true;
    }
    if ((ok == false) && (strcmp(argv[1], "-sinc") == 0)) {
      argc--;
      argv++;
      if (source.GetZ() == 1){
      	interpolator = new irtkSincInterpolateImageFunction2D;
      } else {
      	interpolator = new irtkSincInterpolateImageFunction;
      }
      ok = true;
    }
    if (ok == false) {
      cerr << "Can not parse argument " << argv[1] << endl;
      usage();
    }
  }

  // Set up a mask,
  if (mask_name == NULL) {
    mask.Read(target_name);

    irtkGreyPixel *ptr2mask = mask.GetPointerToVoxels();
    irtkRealPixel *ptr2tgt  = target.GetPointerToVoxels();

    for (int i = 0; i < target.GetNumberOfVoxels(); i++) {
      if (*ptr2tgt > Tp)
        *ptr2mask = 1;
      else
        *ptr2mask = 0;

      ++ptr2tgt;
      ++ptr2mask;
    }
  } else {
    mask.Read(mask_name);
    if ((mask.GetX() != target.GetX()) ||
        (mask.GetY() != target.GetY()) ||
        (mask.GetZ() != target.GetZ())) {
      cerr << "evaluation2: Target and mask dimensions mismatch. Exiting." << endl;
      exit(1);
    }
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

  if (trans_name == 0) {
    transformation = new irtkRigidTransformation;
  } else {
    transformation = irtkTransformation::New(trans_name);
  }

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
    if (histo_name != NULL) {
      histogram.WriteAsImage(histo_name);
    }
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
}
