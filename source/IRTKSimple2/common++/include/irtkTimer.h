/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkTimer.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKTIMER_H

#define _IRTKTIMER_H

#include <time.h>

#ifndef WIN32
#include <sys/times.h>
#endif

// Uncomment to check for normalization of tms_utime, see also <limits.h>
// #include <unistd.h>

// Local includes
#include <irtkObject.h>

class irtkTimer : public irtkObject
{

public:

  /// Constructor
  irtkTimer() {
    // Uncomment to check for normalization of tms_utime //
    // cerr << sysconf(3) << " " << CLK_TCK << endl;
    this->Reset();
  };

  /// Destructor
  virtual ~irtkTimer() {};

  /// Returns used cpu time.
  float Report();

  /// Resets cpu time to negative current time (also used by constrcutor).
  void  Reset();

private:

#ifndef WIN32

  /// Member to hold time in user mode.
  long       _usr_time;

  /// Meber struct to hold cpu time.
  struct tms _cpu_time;

#endif

};

inline float irtkTimer::Report()
{
#ifndef WIN32
  times(&_cpu_time);
  _usr_time += _cpu_time.tms_utime;
  return _usr_time/(float)CLOCKS_PER_SEC;
#else
  return 0;
#endif
}

inline void irtkTimer::Reset()
{
#ifndef WIN32
  times(&_cpu_time);
  _usr_time = -_cpu_time.tms_utime;
#endif
}

#endif
