/*=========================================================================
* GPU accelerated motion compensation for MRI
*
* Copyright (c) 2016 Bernhard Kainz, Amir Alansary, Maria Kuklisova-Murgasova,
* Kevin Keraudren, Markus Steinberger
* (b.kainz@imperial.ac.uk)
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
=========================================================================*/
// runStackSLIC.c: interface for the SLIC superpixel segmentation.
//===========================================================================
// This code is an interface for the superpixel segmentation technique described in:
// "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and 
// Sabine Susstrunk, IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282, November 2012.
//  
// AUTORIGHTS
// Copyright (C) 2015 Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
// Created by Radhakrishna Achanta on 12/06/15.
//
// Modified by: Amir Alansary, 02/08/2015.
//===========================================================================
#ifndef _SLIC
#define _SLIC

#include <irtkImage.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <time.h>  
#include <irtkImageFunction.h>
#include <stdlib.h>

// SLIC libraries 
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef WIN32
typedef unsigned int uint;
#endif

template <typename T>
class runStackSLIC
{
public:

    // runStackSLIC();
    // ~runStackSLIC();

    void rgbtolab(int* rin, int* gin, int* bin, int sz, double* lvec, double* avec, double* bvec);

    void getLABXYSeeds(int STEP, int width, int height, int* seedIndices, int* numseeds);

    void PerformSuperpixelSLIC(double* lvec, double* avec, double* bvec, double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx, double* kseedsy, int width, int height, int numseeds, int* klabels, int STEP, double compactness);

    //===========================================================================
    ///	PerformSuperpixelSLICO
    ///
    /// This function picks the maximum value of color distance as compact factor
    /// M. So there is no need to input a constant value of M and S. There are two
    /// advantages:
    ///
    /// [1] The algorithm now better handles both textured and non-textured regions
    /// [2] There is not need to set any parameters!!!
    ///
    /// SLICO (or SLIC Zero) dynamically varies only the compactness factor,
    /// not the step size S.
    //===========================================================================
    void PerformSuperpixelSLICO(double* lvec, double* avec, double* bvec, double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx, double* kseedsy, int width, int height, int numseeds, int* klabels, int STEP);

    void EnforceSuperpixelConnectivity(int* labels, int width, int height, int numSuperpixels, int* nlabels, int* finalNumberOfLabels);

    //=================================================================================
    /// DrawContoursAroundSegments
    ///
    /// Internal contour drawing option exists. One only needs to comment the if
    /// statement inside the loop that looks at neighborhood.
    //=================================================================================
    void DrawContoursAroundSegments(
            unsigned int*			ubuff,
            const int*				labels,
            const int				  width,
            const int				  height,
            const unsigned int&		color = 0xff0000);

    void DrawContoursAroundSegmentsTwoColors(
            unsigned int*			img,
            const int*				labels,
            const int&				width,
            const int&				height);

    // =========================================================================================
    // =========================================================================================
    /** This module runs SLIC superpixel segmentation.
        Attributes:
                @param compactness:   compactness factor
                @param stacks: 	      input stacks
                @param sStacks: 	  output superpixel stacks
    */
    void segmentSLIC(irtkGenericImage<T> &stack, irtkGenericImage<T> &stack_spx, unsigned int &spx_sz_0, unsigned int &spx_sz_1, unsigned int stackNo, bool _debug);

    // private:
    //   unsigned int  ubuff, img;
    //   int           labels, rin, gin, bin;
    //   double        lvec, avec, bvec, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels;

};

#endif
