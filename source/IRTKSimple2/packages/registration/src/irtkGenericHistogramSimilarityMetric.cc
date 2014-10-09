/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkGenericHistogramSimilarityMetric.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#include <irtkRegistration.h>

double irtkGenericHistogramSimilarityMetric::Evaluate()
{
  // Evaluate similarity from histogram
  switch (_SimilarityMeasure) {
  case K:
    return +this->_histogram->Kappa();
    break;
  case LC:
    return +this->_histogram->LabelConsistency();
    break;
  case JE:
    return -this->_histogram->JointEntropy();
    break;
  case CC:
    return +this->_histogram->CrossCorrelation();
    break;
  case MI:
    return +this->_histogram->MutualInformation();
    break;
  case NMI:
    return +this->_histogram->NormalizedMutualInformation();
    break;
  case SSD:
    return -this->_histogram->SumsOfSquaredDifferences() /
           (double)this->_histogram->NumberOfSamples();
    break;
  case CR_XY:
    return +this->_histogram->CorrelationRatioXY();
    break;
  case CR_YX:
    return +this->_histogram->CorrelationRatioYX();
    break;
  default:
    return 0;
    break;
  }
}
