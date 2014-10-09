
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

#pragma once


#include <irtkImage.h>
#include <irtkImageFunction.h>
#include <irtkHistogram.h>
#include <irtkTransformation.h>

struct EvalResult
{
	double cc;
	double ssd;
	double je;
	double mi;
	double nmi;
	double cr_x;
	double cr_y;
	double lc;
	double ks;
	double psnr;
	double varX;
	double meanX;
	double jentropy;
};


class irtkEvaluation {

public:
	irtkEvaluation(irtkRealImage target_, irtkRealImage source_) : target(target_), source(source_) 
	{
		nbins_x = 0;
		nbins_y = 0;
	};
	~irtkEvaluation(){};

	EvalResult evaluate();

private:
	irtkRealImage target;
	irtkRealImage source;
	int nbins_x, nbins_y;
};