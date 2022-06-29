
#ifndef SCINV_CHI_PHPP_H
#define SCINV_CHI_PHPP_H

#include <Eigen/Core>
#include "vertex.h"
#include "single_freq_gf.h"
#include "mc_vertex.h"

namespace SCinverter {
namespace nambu {

/// Class for the all dense matrix chi.
class chi : public vertex<VertMatDim> {
public:
  // Constructor
  chi(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
      int size = BlockNum) : vertex<VertMatDim>(p, ks, channel, p["NOMEGA4"], size) {};

  virtual ~chi() {};

  using vertex<VertMatDim>::operator();

  inline const Eigen::Block<MatrixConstMap<complex> > operator()(int type, int K1, int K2, int ind1, int ind2) const {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      throw std::runtime_error("chi exceed block number");
    }

    if (type == 0 || type == 1) {
      int shift1 = K1 * 2 * n_omega4_;
      int shift2 = K2 * 2 * n_omega4_;
      int omega_size = 2 * n_omega4_;

      return MatrixConstMap<complex>(vert_(type, flatten_ind(ind1, ind2)).data(),
                                     mat_size_, mat_size_).block(shift1, shift2, omega_size, omega_size);
    }

    throw std::runtime_error("chi type unreachable");
  }
};

class chi_phpp : public chi {
public:
  // Constructor
  chi_phpp(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
           int size = BlockNum) : chi(p, ks, channel, size) {};

  virtual ~chi_phpp() {};
};

/// This is a class for the CLUSTER susceptibility.
class chi_cluster_phpp : public chi_phpp {
public:
  // Constructor
  chi_cluster_phpp(const alps::params &p, const k_space_structure &ks,
                   vert_channel_enum channel, int size = BlockNum) : chi_phpp(p, ks, channel, size) {};

  /// get the susceptibility out of the green's function and the Monte Carlo vertex
  void compute_chi(alps::hdf5::archive &ar_in, const fermionic_green_function &g2);

  void read(alps::hdf5::archive &ar) { chi::read(ar, "chi_cluster"); };
  void write(alps::hdf5::archive &ar) { chi::write(ar, "chi_cluster"); };
};

// LATTICE susceptibility with cluster full vertex F and lattice chi0
class chi_lattice_phpp : public chi_phpp {
public:
  // Constructor
  chi_lattice_phpp(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                     int size = BlockNum) : chi_phpp(p, ks, channel, size) {};

  void read(alps::hdf5::archive &ar) { chi::read(ar, "chi_lattice"); };
  void write(alps::hdf5::archive &ar) { chi::write(ar, "chi_lattice"); };
};

/// This is a class for the coarse-grained LATTICE susceptibility with cluster irreducible vertex \Gamma and lattice chi0.
/// In the Jarrell literature it is usually called \f$ \overline{\chi} \f$
class chi_cglattice_phpp : public chi_phpp {
public:
  // Constructor
  chi_cglattice_phpp(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                     int size = 1) : chi_phpp(p, ks, channel, size) {};

  void read(alps::hdf5::archive &ar) { chi::read(ar, "chi_cglattice"); };
  void write(alps::hdf5::archive &ar) { chi::write(ar, "chi_cglattice"); };
};

} // namespace nambu
} // namespace SCinverter

#endif //SCINV_CHI_PHPP_H
