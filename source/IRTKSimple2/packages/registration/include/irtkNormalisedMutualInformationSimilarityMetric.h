/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkNormalisedMutualInformationSimilarityMetric.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKNORMALISEDMUTUALINFORMATIONSIMILARITYMETRIC_H

#define _IRTKNORMALISEDMUTUALINFORMATIONSIMILARITYMETRIC_H

/**
 * Class for voxel similarity measure based on mutual information
 *
 */

class irtkNormalisedMutualInformationSimilarityMetric : public irtkHistogramSimilarityMetric
{

public:

  /// Constructor
  irtkNormalisedMutualInformationSimilarityMetric(int = 64, int = 64);

  /// Evaluate similarity measure
  virtual double Evaluate();

};

inline irtkNormalisedMutualInformationSimilarityMetric::irtkNormalisedMutualInformationSimilarityMetric(int nbins_x, int nbins_y) : irtkHistogramSimilarityMetric (nbins_x, nbins_y)
{
}

inline double irtkNormalisedMutualInformationSimilarityMetric::Evaluate()
{
  return _histogram->NormalizedMutualInformation();
}

#endif
