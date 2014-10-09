/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkErosion.h 358 2011-06-28 07:10:17Z pa100 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2011-06-28 08:10:17 +0100 (Tue, 28 Jun 2011) $
  Version   : $Revision: 358 $
  Changes   : $Author: pa100 $

=========================================================================*/

#ifndef _IRTKEROSION_H

#define _IRTKEROSION_H

#include <irtkImageToImage.h>

/**
 * Class for erosion of images
 *
 * This class defines and implements the morphological erosion of images.
 *
 */

template <class VoxelType> class irtkErosion : public irtkImageToImage<VoxelType>
{

protected:

  /// Returns whether the filter requires buffering
  virtual bool RequiresBuffering();

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Initialize the filter
  virtual void Initialize();

  /// What connectivity to assume when running the filter.
  irtkConnectivityType _Connectivity;

  // List of voxel offsets of the neighbourhood.
  irtkNeighbourhoodOffsets _offsets;

public:

  /// Constructor
  irtkErosion();

  /// Destructor
  ~irtkErosion();

  /// Run erosion
  virtual void Run();

  SetMacro(Connectivity, irtkConnectivityType);

  GetMacro(Connectivity, irtkConnectivityType);
};

#endif
