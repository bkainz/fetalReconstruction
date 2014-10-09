/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: newt2.h 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

extern void newt2(float x[], int n, int *check, void (*vecfunc)(int, float [], float []));

extern double x_invert, y_invert, z_invert;

extern irtkTransformation *irtkTransformationPointer;

extern void irtkTransformationEvaluate(int, float point[], float fval[]);
