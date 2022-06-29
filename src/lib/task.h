
#ifndef SCINV_TASK_H
#define SCINV_TASK_H

#include "k_space_structure.h"

namespace SCinverter {

class task {
public:
  task(const alps::params &p): p_(p), ks_(p),
                               ar_(p["SIM_FILE_NAME"], "r"),
                               out_file_basename_(p["OUT_FILE_NAME"].as<std::string>()),
                               U_(p["U"]), beta_(p["BETA"]), mu_(p["MU"]),
                               n_sites_(p["dca.SITES"]), n_freq_(p["NMATSUBARA"]) {};

  virtual ~task(){};

protected:
  /// ALPS parameter file that contains all the important parameters
  const alps::params &p_;
  /// k-space information: adding and subtracting momenta, k-points, etc.
  const k_space_structure ks_;
  /// hdf5 input file
  alps::hdf5::archive ar_;
  /// base name for output files generated throughout the code
  std::string out_file_basename_;

  /// interaction strength
  double U_;
  /// inverse temperature
  double beta_;
  /// number of cluster sites
  int n_sites_;
  /// number of matsubara frequencies
  int n_freq_;
  /// chemical potential
  double mu_;
};

} // namespace SCinverter

#endif //SCINV_TASK_H
