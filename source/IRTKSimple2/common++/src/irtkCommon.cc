/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkCommon.cc 880 2013-05-16 11:53:39Z as12312 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2013-05-16 12:53:39 +0100 (Thu, 16 May 2013) $
  Version   : $Revision: 880 $
  Changes   : $Author: as12312 $

=========================================================================*/

#include <irtkCommon.h>

void PrintVersion(ostream &out, const char* revisionString)
{
	out << "(SVN revision: ";
	// Extract the number from the SVN supplied revision string.
	int len = strlen(revisionString);
	const char *ptr = revisionString;
	for (int i = 0; i < len; ++i){
		if (*ptr >= '0' && *ptr <= '9')
				out << *ptr;
		++ptr;
	}
	out << ")\n";

}

