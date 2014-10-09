# This file is configured by CMake automatically as DartConfiguration.tcl
# If you choose not to use CMake, this file may be hand configured, by
# filling in the required variables.


# Configuration directories and files
SourceDirectory: E:/FetalReconstruction/source/IRTKSimple2
BuildDirectory: E:/FetalReconstruction/source/IRTKSimple2

# Where to place the cost data store
CostDataFile: 

# Site is something like machine.domain, i.e. pragmatic.crd
Site: STEINBER-TUGRAZ

# Build name is osname-revision-compiler, i.e. Linux-2.4.2-2smp-c++
BuildName: Win32-vs11

# Submission information
IsCDash: TRUE
CDashVersion: 
QueryCDashVersion: 
DropSite: itk.doc.ic.ac.uk
DropLocation: /vol/vipcvs/Dart/projectincoming
DropSiteUser: 
DropSitePassword: 
DropSiteMode: 
DropMethod: scp
TriggerSite: http://www.doc.ic.ac.uk/~rc3/Dashboard/TestingResults/Dart.pl
ScpCommand: SCPCOMMAND-NOTFOUND

# Dashboard start time
NightlyStartTime: 00:00:00 EST

# Commands for the build/test/submit cycle
ConfigureCommand: "D:/Program Files (x86)/CMake 2.8/bin/cmake.exe" "E:/FetalReconstruction/source/IRTKSimple2"
MakeCommand: C:\PROGRA~2\MICROS~1.0\Common7\IDE\devenv.com IRTK.sln /build ${CTEST_CONFIGURATION_TYPE} /project ALL_BUILD
DefaultCTestConfigurationType: Release

# CVS options
# Default is "-d -P -A"
CVSCommand: CVSCOMMAND-NOTFOUND
CVSUpdateOptions: -d -A -P

# Subversion options
SVNCommand: d:/Program Files/TortoiseSVN/bin/svn.exe
SVNOptions: 
SVNUpdateOptions: 

# Git options
GITCommand: GITCOMMAND-NOTFOUND
GITUpdateOptions: 
GITUpdateCustom: 

# Generic update command
UpdateCommand: 
UpdateOptions: 
UpdateType: 

# Compiler info
Compiler: C:/Program Files (x86)/Microsoft Visual Studio 11.0/VC/bin/x86_amd64/cl.exe

# Dynamic analysis (MemCheck)
PurifyCommand: 
ValgrindCommand: 
ValgrindCommandOptions: 
MemoryCheckCommand: MEMORYCHECK_COMMAND-NOTFOUND
MemoryCheckCommandOptions: 
MemoryCheckSuppressionFile: 

# Coverage
CoverageCommand: d:/strawberry/c/bin/gcov.exe
CoverageExtraFlags: -l

# Cluster commands
SlurmBatchCommand: SLURM_SBATCH_COMMAND-NOTFOUND
SlurmRunCommand: SLURM_SRUN_COMMAND-NOTFOUND

# Testing options
# TimeOut is the amount of time in seconds to wait for processes
# to complete during testing.  After TimeOut seconds, the
# process will be summarily terminated.
# Currently set to 25 minutes
TimeOut: 1500

UseLaunchers: 0
CurlOptions: 
# warning, if you add new options here that have to do with submit,
# you have to update cmCTestSubmitCommand.cxx

# For CTest submissions that timeout, these options
# specify behavior for retrying the submission
CTestSubmitRetryDelay: 5
CTestSubmitRetryCount: 3
