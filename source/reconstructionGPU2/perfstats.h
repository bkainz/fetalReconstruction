/*=========================================================================
Library   : Image Registration Toolkit (IRTK)
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : $Date: 2013-11-15 14:36:30 +0100 (Fri, 15 Nov 2013) $
Version   : $Revision: 1 $
Changes   : $Author: bkainz $

Copyright (c) 2014, Bernhard Kainz, Markus Steinberger,
Maria Murgasova, Kevin Keraudren
All rights reserved.

If you use this work for research we would very much appreciate if you cite
Bernhard Kainz, Markus Steinberger, Maria Kuklisova-Murgasova, Christina Malamateniou,
Wolfgang Wein, Thomas Torsney-Weir, Torsten Moeller, Mary Rutherford,
Joseph V. Hajnal and Daniel Rueckert:
Fast Volume Reconstruction from Motion Corrupted 2D Slices.
IEEE Transactions on Medical Imaging, in press, 2015

IRTK IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN
AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE
TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE
CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED
HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

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
=========================================================================*/

#ifndef PERFSTATS_H
#define PERFSTATS_H

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <ctime>

//need better time count for full evaluation (guess double is not long enough)
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
namespace pt = boost::posix_time;

struct PerfStats {
  enum Type { TIME, COUNT, PERCENTAGE };
  struct Stats {
    std::vector<double> data;
    Type type;
    double sum() const { return std::accumulate(data.begin(), data.end(), 0.0); }
    double average() const { return sum() / std::max(data.size(), size_t(1)); }
    double max() const { return *std::max_element(data.begin(), data.end()); }
    double min() const { return *std::min_element(data.begin(), data.end()); }
  };

  std::map<std::string, Stats> stats;
  pt::ptime last;

  static pt::ptime get_time() {
    return pt::microsec_clock::local_time();
  }

  void sample(const std::string& key, double t, Type type = COUNT) {
    Stats& s = stats[key];
    s.data.push_back(t);
    s.type = type;
  }
  pt::ptime start(void){
    last = get_time();
    return last;
  }
  pt::ptime sample(const std::string &key){
    const pt::ptime now = get_time();
    pt::time_duration diff = now - last;
    sample(key, diff.total_milliseconds() / 1000.0, TIME);
    last = now;
    return now;
  }
  const Stats& get(const std::string& key) const { return stats.find(key)->second; }
  void reset(void) { stats.clear(); }
  void reset(const std::string & key);
  void print(std::ostream& out = std::cout) const;
};

inline void PerfStats::reset(const std::string & key){
  std::map<std::string, Stats>::iterator s = stats.find(key);
  if (s != stats.end())
    s->second.data.clear();
}

inline void PerfStats::print(std::ostream& out) const {
  std::cout.precision(10);
  for (std::map<std::string, Stats>::const_iterator it = stats.begin(); it != stats.end(); it++){
    out << it->first << ":";
    out << std::string("\t\t\t").substr(0, 3 - ((it->first.size() + 1) >> 3));
    switch (it->second.type){
    case TIME: {
      out << it->second.average()*1000.0 << " ms" << "\t(max = " << it->second.max() * 1000 << " ms" << ")\n";
    } break;
    case COUNT: {
      out << it->second.average() << " ms" << "\t(max = " << it->second.max() << " ms" << " )\n";
    } break;
    case PERCENTAGE: {
      out << it->second.average()*100.0 << " ms" << "%\t(max = " << it->second.max() * 100 << " ms" << " %)\n";
    } break;
    }
  }
}

extern PerfStats Stats;

#endif // PERFSTATS_H
