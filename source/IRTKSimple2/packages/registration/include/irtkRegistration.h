/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkRegistration.h 904 2013-06-05 14:56:45Z sp2010 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2013-06-05 15:56:45 +0100 (Wed, 05 Jun 2013) $
  Version   : $Revision: 904 $
  Changes   : $Author: sp2010 $

=========================================================================*/

#ifndef _IRTKREGISTRATION_H

#define _IRTKREGISTRATION_H

#define MAX_NO_RESOLUTIONS 10

// Definition of available states for individual DOFs
typedef enum { Active, Passive } DOFStatus;

// Definition of available optimization m
typedef enum { DownhillDescent,
               GradientDescent,
               GradientDescentConstrained,
               SteepestGradientDescent,
               ConjugateGradientDescent,
               ClosedForm
             } irtkOptimizationMethod;

// Definition of available similarity measures
typedef enum { JE, CC, MI, NMI, SSD, CR_XY, CR_YX, LC, K, ML, NGD, NGP, NGS }
irtkSimilarityMeasure;

#include <irtkImage.h>

#include <irtkHistogram.h>

#include <irtkResampling.h>

#include <irtkImageFunction.h>

#include <irtkTransformation.h>

#include <irtkSimilarityMetric.h>

#include <irtkOptimizer.h>

#include <irtkUtil.h>

class irtkRegistration : public irtkObject
{

public:

  irtkHistory *history;
  double last_similarity;

  /// Evaluate similarity metric
  virtual double Evaluate() = 0;

  /// Evaluate gradient of similarity metric
  virtual double EvaluateGradient(float, float *) = 0;

};

double combine_mysimilarity(double,double,double,double);
double combine_mysimilarity(irtkSimilarityMetric **, double *, double);

#include <irtkPointRegistration.h>
//#include <irtkSurfaceRegistration.h>
//#include <irtkModelRegistration.h>
#include <irtkImageRegistration.h>
#include <irtkSymmetricImageRegistration.h>

#endif
