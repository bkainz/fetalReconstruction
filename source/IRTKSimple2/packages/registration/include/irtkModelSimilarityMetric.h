/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkModelSimilarityMetric.h 665 2012-08-29 21:45:17Z ws207 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2009 onwards
  Date      : $Date: 2012-08-29 22:45:17 +0100 (Wed, 29 Aug 2012) $
  Version   : $Revision: 665 $
  Changes   : $Author: ws207 $

=========================================================================*/

#ifndef _IRTKMODELSIMILARITYMETRIC_H

#define _IRTKMODELSIMILARITYMETRIC_H

#ifdef HAS_VTK

/**
 * Generic class for model similarity measures.
 *
 */

class irtkModelSimilarityMetric : public irtkObject
{

protected:
	
	/// Target image
	irtkGreyImage *_image;
		
public:

  /// Constructor
  irtkModelSimilarityMetric(irtkGreyImage *);

  /// Destructor
  virtual ~irtkModelSimilarityMetric();

  /// Evaluate metric
  virtual double Evaluate(double *, double * = NULL, double * = NULL) = 0;

};

inline irtkModelSimilarityMetric::irtkModelSimilarityMetric(irtkGreyImage *image)
{
	_image  = image;
}

inline irtkModelSimilarityMetric::~irtkModelSimilarityMetric()
{
}

#include <irtkModelGradientSimilarityMetric.h>
#include <irtkModelCorrelationSimilarityMetric.h>

#endif

#endif
