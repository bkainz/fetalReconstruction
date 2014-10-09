/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkCorrelationRatioYXSimilarityMetric.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKCORRELATIONRATIOYXSIMILARITYMETRIC_H

#define _IRTKCORRELATIONRATIOYXSIMILARITYMETRIC_H

/**
 * Class for voxel similarity measure based on correlation ratio
 *
 */

class irtkCorrelationRatioYXSimilarityMetric : public irtkHistogramSimilarityMetric
{

public:

  /// Constructor
  irtkCorrelationRatioYXSimilarityMetric(int = 64, int = 64);

  /// Evaluate similarity measure
  virtual double Evaluate();

};

inline irtkCorrelationRatioYXSimilarityMetric::irtkCorrelationRatioYXSimilarityMetric(int nbins_x, int nbins_y) : irtkHistogramSimilarityMetric (nbins_x, nbins_y)
{
}

inline double irtkCorrelationRatioYXSimilarityMetric::Evaluate()
{
  return _histogram->CorrelationRatioYX();
}

#endif
