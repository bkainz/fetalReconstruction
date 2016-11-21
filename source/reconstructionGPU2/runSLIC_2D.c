// runSLIC_2D.cpp: interface for the SLIC superpixel segmentation.
//===========================================================================
// This code is an interface for the superpixel segmentation technique described in:
// "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and 
// Sabine Susstrunk, IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282, November 2012.
//
// Modified by: Amir Alansary, June 2015.
//===========================================================================
#include <irtkImage.h>
#include <vector>
#include <string>
#include <perfstats.h>
#include <fstream>
#include <iostream>
#include <time.h>  
#include <irtkImageFunction.h>
#include <stdlib.h>

// SLIC libraries 
#include <stdio.h>
#include <math.h>
#include <float.h>

void rgbtolab(int* rin, int* gin, int* bin, int sz, double* lvec, double* avec, double* bvec)
{
    int i; int sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    const double epsilon = 0.008856;	//actual CIE standard
    const double kappa   = 903.3;		//actual CIE standard
    
    const double Xr = 0.950456;	//reference white
    const double Yr = 1.0;		//reference white
    const double Zr = 1.088754;	//reference white
    double xr,yr,zr;
    double fx, fy, fz;
    double lval,aval,bval;
    
    for(int i = 0; i < sz; i++)
    {
        sR = rin[i]; sG = gin[i]; sB = bin[i];
        R = sR/255.0;
        G = sG/255.0;
        B = sB/255.0;
        
        if(R <= 0.04045)	r = R/12.92;
        else				r = pow((R+0.055)/1.055,2.4);
        if(G <= 0.04045)	g = G/12.92;
        else				g = pow((G+0.055)/1.055,2.4);
        if(B <= 0.04045)	b = B/12.92;
        else				b = pow((B+0.055)/1.055,2.4);
        
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        
        //------------------------
        // XYZ to LAB conversion
        //------------------------
        xr = X/Xr;
        yr = Y/Yr;
        zr = Z/Zr;
        
        if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        else				fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        else				fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        else				fz = (kappa*zr + 16.0)/116.0;
        
        lval = 116.0*fy-16.0;
        aval = 500.0*(fx-fy);
        bval = 200.0*(fy-fz);
        
        lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
    }
}

void getLABXYSeeds(int STEP, int width, int height, int* seedIndices, int* numseeds)
{
    const bool hexgrid = false;
    int n;
    int xstrips, ystrips;
    int xerr, yerr;
    double xerrperstrip,yerrperstrip;
    int xoff,yoff;
    int x,y;
    int xe,ye;
    int seedx,seedy;
    int i;

    xstrips = (0.5+(double)(width)/(double)(STEP));
    ystrips = (0.5+(double)(height)/(double)(STEP));
    
    xerr = width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = width - STEP*xstrips;}
    yerr = height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = height- STEP*ystrips;}
    
    xerrperstrip = (double)(xerr)/(double)(xstrips);
    yerrperstrip = (double)(yerr)/(double)(ystrips);
    
    xoff = STEP/2;
    yoff = STEP/2;
    
    n = 0;
    for( y = 0; y < ystrips; y++ )
    {
        ye = y*yerrperstrip;
        for( x = 0; x < xstrips; x++ )
        {
            xe = x*xerrperstrip;
            seedx = (x*STEP+xoff+xe);
            if(hexgrid){ seedx = x*STEP+(xoff<<(y&0x1))+xe; if(seedx >= width)seedx = width-1; }//for hex grid sampling
            seedy = (y*STEP+yoff+ye);
            i = seedy*width + seedx;
            seedIndices[n] = i;
            n++;
        }
    }
    *numseeds = n;
}

void PerformSuperpixelSLIC(double* lvec, double* avec, double* bvec, double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx, double* kseedsy, int width, int height, int numseeds, int* klabels, int STEP, double compactness)
{
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;
    int itr;
    int n;
    int x,y;
    int i;
    int ind;
    int r,c;
    int k;
    int sz = width*height;
    const int numk = numseeds;
    int offset = STEP;
    
    double* clustersize = new double[numk];
    double* inv         = new double[numk];
    double* sigmal      = new double[numk];
    double* sigmaa      = new double[numk];
    double* sigmab      = new double[numk];
    double* sigmax      = new double[numk];
    double* sigmay      = new double[numk];
    double* distvec     = new double[sz];
    double 	invwt 		= 1.0/((STEP/compactness)*(STEP/compactness));
    
    for( itr = 0; itr < 10; itr++ )
    {
        for(i = 0; i < sz; i++){distvec[i] = DBL_MAX;}

        for( n = 0; n < numk; n++ )
        {
            x1 = kseedsx[n]-offset; if(x1 < 0) x1 = 0;
            y1 = kseedsy[n]-offset; if(y1 < 0) y1 = 0;
            x2 = kseedsx[n]+offset; if(x2 > width)  x2 = width;
            y2 = kseedsy[n]+offset; if(y2 > height) y2 = height;
            
            for( y = y1; y < y2; y++ )
            {
                for( x = x1; x < x2; x++ )
                {
                    i = y*width + x;
                    
                    l = lvec[i];
                    a = avec[i];
                    b = bvec[i];
                    
                    dist =			(l - kseedsl[n])*(l - kseedsl[n]) +
                            (a - kseedsa[n])*(a - kseedsa[n]) +
                            (b - kseedsb[n])*(b - kseedsb[n]);
                    
                    distxy =		(x - kseedsx[n])*(x - kseedsx[n]) + (y - kseedsy[n])*(y - kseedsy[n]);

                    dist += distxy*invwt;
                    
                    if(dist < distvec[i])
                    {
                        distvec[i] = dist;
                        klabels[i]  = n;
                    }
                }
            }
        }
        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        for(k = 0; k < numk; k++)
        {
            sigmal[k] = 0;
            sigmaa[k] = 0;
            sigmab[k] = 0;
            sigmax[k] = 0;
            sigmay[k] = 0;
            clustersize[k] = 0;
        }
        
        ind = 0;
        for( r = 0; r < height; r++ )
        {
            for( c = 0; c < width; c++ )
            {
                if(klabels[ind] >= 0)
                {
                    sigmal[klabels[ind]] += lvec[ind];
                    sigmaa[klabels[ind]] += avec[ind];
                    sigmab[klabels[ind]] += bvec[ind];
                    sigmax[klabels[ind]] += c;
                    sigmay[klabels[ind]] += r;
                    clustersize[klabels[ind]] += 1.0;
                }
                ind++;
            }
        }
        
        {for( k = 0; k < numk; k++ )
            {
                if( clustersize[k] <= 0 ) clustersize[k] = 1;
                inv[k] = 1.0/clustersize[k];//computing inverse now to multiply, than divide later
            }}

        {for( k = 0; k < numk; k++ )
            {
                kseedsl[k] = sigmal[k]*inv[k];
                kseedsa[k] = sigmaa[k]*inv[k];
                kseedsb[k] = sigmab[k]*inv[k];
                kseedsx[k] = sigmax[k]*inv[k];
                kseedsy[k] = sigmay[k]*inv[k];
            }}
    }

    if(sigmal) 		delete [] sigmal;
    if(sigmaa) 		delete [] sigmaa;
    if(sigmab) 		delete [] sigmab;
    if(sigmax) 		delete [] sigmax;
    if(sigmay) 		delete [] sigmay;
    if(clustersize) delete [] clustersize;
    if(inv) 		delete [] inv;
    if(distvec) 	delete [] distvec;
}

void EnforceSuperpixelConnectivity(int* labels, int width, int height, int numSuperpixels,int* nlabels, int* finalNumberOfLabels)
{
    int i,j,k;
    int n,c,count;
    int x,y;
    int ind;
    int label;
    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};
    const int sz = width*height;
    const int SUPSZ = sz/numSuperpixels;
    int* xvec = new int[SUPSZ*10];
    int* yvec = new int[SUPSZ*10];

    for( i = 0; i < sz; i++ ) nlabels[i] = -1;
    int oindex = 0;
    int adjlabel = 0;//adjacent label
    label = 0;
    for( j = 0; j < height; j++ )
    {
        for( k = 0; k < width; k++ )
        {
            if( 0 > nlabels[oindex] )
            {
                nlabels[oindex] = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                {for( n = 0; n < 4; n++ )
                    {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                        {
                            int nindex = y*width + x;
                            if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
                        }
                    }}
                
                count = 1;
                for( c = 0; c < count; c++ )
                {
                    for( n = 0; n < 4; n++ )
                    {
                        x = xvec[c] + dx4[n];
                        y = yvec[c] + dy4[n];
                        
                        if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                        {
                            int nindex = y*width + x;
                            
                            if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }
                        
                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if(count <= SUPSZ >> 2)
                {
                    for( c = 0; c < count; c++ )
                    {
                        ind = yvec[c]*width+xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    *finalNumberOfLabels = label;
    
    if(xvec) delete [] xvec;
    if(yvec) delete [] yvec;

}

// =========================================================================================
// =========================================================================================

void runSLIC_2D(float noLabels, double m, irtkRealImage &stack, vector<irtkRealImage> &sStacks)
{
    /* This module runs SLIC superpixel segmentation.
                Attributes:
                        noLabels:   desired number of superpixels
                        m: 			compactness factor
                        stacks: 	input stacks
                        sStacks: 	output superpixel stacks
        */

    //---------------------------
    // Variable default declarations
    //---------------------------
    int numSuperpixels = 200;//default value
    double compactness = 10;//default value
    int width;
    int height;
    int sz;
    int* rin; int* gin; int* bin;
    int* klabels;
    int* clabels;
    double* lvec; double* avec; double* bvec;
    int step;
    int* seedIndices;
    int numseeds;
    double* kseedsx;double* kseedsy;
    double* kseedsl;double* kseedsa;double* kseedsb;
    int finalNumberOfLabels;
    char savename[256];

    // extract stacks by selecting the appropriate region of the stack
    irtkRealImage 	sStack = stack;
    // image attributes contain image and voxel size
    irtkImageAttributes attr   = stack.GetImageAttributes();
    // get first image dimensions and pointer
    width  		= (int)attr._x;
    height 		= (int)attr._y;
    sz 			= width*height;
    //---------------------------
    numSuperpixels  = (int)(m * sqrt(sz/2));
    compactness     = m;

    cout << "numSuperpixels " << numSuperpixels << endl;
    cout << "Compactness "    << compactness    << endl;
    // loop all the 2D slices in each stack
    cout << "Number of slices: " << attr._z << endl;
    //attr._z is number of slices in the stack
    for (int z = 0; z < attr._z; z++)
    {
        //create slice by selecting the appropriate region of the stack
        irtkRealImage slice   = stack.GetRegion(0, 0, z, attr._x, attr._y, z + 1);
        //---------------------------
        // Allocate memory
        //---------------------------
        rin    	    = new int[sz];
        gin    	    = new int[sz];
        bin    	    = new int[sz];
        lvec    	= new double[sz];
        avec    	= new double[sz];
        bvec    	= new double[sz];
        klabels 	= new int[sz];//original k-means labels
        clabels 	= new int[sz];//corrected labels after enforcing connectivity
        seedIndices = new int[sz];
        // copy image contents to a buffer before segmenting color conversion from gray to lab
        // first convert gray to RGB, then RGB to LAB (gives better results than just assigns gray values to LAB)
        int p = 0;
        for (int x = 0; x < attr._x; ++x)
        {
            for (int y = 0; y < attr._y; ++y)
            {
                rin[p] = 0.2989 * slice(x, y, 0);
                gin[p] = 0.5870 * slice(x, y, 0);
                bin[p] = 0.1140 * slice(x, y, 0);
                p++;
            }
        }
        rgbtolab(rin,gin,bin,sz,lvec,avec,bvec);

        //---------------------------
        // Find seeds
        //---------------------------
        step = sqrt((double)(sz)/(double)(numSuperpixels))+0.5;
        getLABXYSeeds(step,width,height,seedIndices,&numseeds);
        kseedsx    = new double[numseeds];
        kseedsy    = new double[numseeds];
        kseedsl    = new double[numseeds];
        kseedsa    = new double[numseeds];
        kseedsb    = new double[numseeds];
        for(int k = 0; k < numseeds; k++)
        {
            kseedsx[k] = seedIndices[k]%width;
            kseedsy[k] = seedIndices[k]/width;
            kseedsl[k] = lvec[seedIndices[k]];
            kseedsa[k] = avec[seedIndices[k]];
            kseedsb[k] = bvec[seedIndices[k]];
        }
        //---------------------------
        // Compute superpixels
        //---------------------------
        PerformSuperpixelSLIC(lvec, avec, bvec, kseedsl,kseedsa,kseedsb,kseedsx,kseedsy,width,height,numseeds,klabels,step,compactness);
        //---------------------------
        // Enforce connectivity
        //---------------------------
        EnforceSuperpixelConnectivity(klabels,width,height,numSuperpixels,clabels,&finalNumberOfLabels);
        //---------------------------
        // Assign output labels
        //---------------------------
        p = 0;
        for (int x = 0; x < attr._x; ++x)
        {
            for (int y = 0; y < attr._y; ++y)
            {
                sStack.Put(x, y, z, clabels[p]);
                p++;
            }
        }

        //---------------------------
        // Deallocate memory
        //---------------------------
        if(rin)  delete [] rin;
        if(gin)  delete [] gin;
        if(bin)  delete [] bin;
        if(lvec) delete [] lvec;
        if(avec) delete [] avec;
        if(bvec) delete [] bvec;
        if(klabels) delete [] klabels;
        if(clabels) delete [] clabels;
        if(seedIndices) delete [] seedIndices;
        if(kseedsx) delete [] kseedsx;
        if(kseedsy) delete [] kseedsy;
        if(kseedsl) delete [] kseedsl;
        if(kseedsa) delete [] kseedsa;
        if(kseedsb) delete [] kseedsb;

    }
    // store the segmented superpixel stack
    sStacks.push_back(sStack);
    cout << "Done.. " << sStacks.size() << endl;
    // // save superpixel output
    // sprintf(savename, "spx.nii.gz");
    // sStack.Write(savename);

}
