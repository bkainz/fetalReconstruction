/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkCorrelationRatioXYSimilarityMetric.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKCORRELATIONRATIOXYSIMILARITYMETRIC_H

#define _IRTKCORRELATIONRATIOXYSIMILARITYMETRIC_H

/**
 * Class for voxel similarity measure based on correlation ratio
 *
 */

class irtkCorrelationRatioXYSimilarityMetric : public irtkHistogramSimilarityMetric
{

public:

  /// Constructor
  irtkCorrelationRatioXYSimilarityMetric(int = 64, int = 64);

  /// Evaluate similarity measure
  virtual double Evaluate();

};

inline irtkCorrelationRatioXYSimilarityMetric::irtkCorrelationRatioXYSimilarityMetric(int nbins_x, int nbins_y) : irtkHistogramSimilarityMetric (nbins_x, nbins_y)
{
}

inline double irtkCorrelationRatioXYSimilarityMetric::Evaluate()
{
  return _histogram->CorrelationRatioXY();
}

#endif
