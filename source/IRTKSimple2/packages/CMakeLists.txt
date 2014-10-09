#SUBDIRS(transformation segmentation registration registration2 rview applications)
#SUBDIRS(transformation segmentation registration)
SUBDIRS(transformation registration)

IF (BUILD_CARDIAC)
  SUBDIRS(cardiac)
ENDIF (BUILD_CARDIAC)


