/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkParallel.cc 880 2013-05-16 11:53:39Z as12312 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2013-05-16 12:53:39 +0100 (Thu, 16 May 2013) $
  Version   : $Revision: 880 $
  Changes   : $Author: as12312 $

=========================================================================*/

#include <irtkParallel.h>

// Default: No debugging of execution time
int debug_time = 0;

// Default: No debugging of TBB code
int tbb_debug = false;

// Default: Number of threads is determined automatically
#ifdef HAS_TBB
int tbb_no_threads = task_scheduler_init::automatic;
#else
int tbb_no_threads = 1;
#endif
