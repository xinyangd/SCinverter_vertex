
#ifndef SCINV_CHI0_PHPP_H
#define SCINV_CHI0_PHPP_H

#include "vertex.h"
#include "single_freq_gf.h"

namespace SCinverter {
namespace nambu {

/// Class for the  \f$ \chi_0\f$ (bare susceptibility) vertex in particle hole-particle particle notation
class chi0_phpp : public vertex<VertVecDim> {
public:
  // Constructor
  chi0_phpp(const alps::params &p, const k_space_structure &ks,
            vert_channel_enum channel, int nomega4, int size = BlockNum) : vertex(p, ks, channel, nomega4, size) {}

  virtual void init_random() {
    vertex<VertVecDim>::init_random();

    for (int K = 0; K < n_sites_; ++K) {
      for (int omega = -n_omega4_; omega < n_omega4_; ++omega) {
        get_ud_from_uu(K, omega);
      }
    }
  }

  virtual ~chi0_phpp() {};

  using vertex<VertVecDim>::operator();

  inline VectorMap<complex> operator()(int type, int K, int ind1, int ind2) {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      throw std::runtime_error("chi0 exceed block number");
    }

    if (type == 0 || type == 1) {
      int shift = K * 2 * n_omega4_;
      return VectorMap<complex>(vert_(type, flatten_ind(ind1, ind2)).data() + shift, 2 * n_omega4_);
    }

    throw std::runtime_error("chi0 type unreachable");
  }

  inline VectorConstMap<complex> operator()(int type, int K, int ind1, int ind2) const {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      throw std::runtime_error("chi0 exceed block number");
    }

    if (type == 0 || type == 1) {
      int shift = K * 2 * n_omega4_;
      return VectorConstMap<complex>(vert_(type, flatten_ind(ind1, ind2)).data() + shift, 2 * n_omega4_);
    }

    throw std::runtime_error("chi0 type unreachable");
  }

  inline complex &operator()(int type, int K, int omega, int ind1, int ind2) {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      throw std::runtime_error("vert element exceed block number");
    }
    if (type == 0 || type == 1)
      return vert_(type, flatten_ind(ind1, ind2), findex(K, omega));

    throw std::runtime_error("vertex element type unreachable");
  }

  inline const complex operator()(int type, int K, int omega, int ind1, int ind2) const {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      return complex(0, 0);
    }
    if (type == 0 || type == 1)
      return vert_(type, flatten_ind(ind1, ind2), findex(K, omega));

    throw std::runtime_error("vertex element type unreachable");
  }

protected:
  // compute chi0 ud from uu use pattern of chi0 matrix, only works for ph channel now
  void get_ud_from_uu(int K, int omega);
};

/// Class for the bare cluster \f$ \chi_0\f$ (bare susceptibility) vertex in particle hole-particle particle notation
class chi0_cluster_phpp : public chi0_phpp {
public:
  // Constructor
  chi0_cluster_phpp(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                    int nomega4, int size = BlockNum) : chi0_phpp(p, ks, channel, nomega4, size) {}

  /// get chi0 out of the Green's function
  void compute_chi0(const fermionic_green_function &g2);

  void read(alps::hdf5::archive &ar) { chi0_phpp::read(ar, "chi0_cluster"); };
  void write(alps::hdf5::archive &ar) { chi0_phpp::write(ar, "chi0_cluster"); };
};

/// Class for the bare coarse-grained lattice \f$ \overline\chi_0\f$ (bare susceptibility) vertex
/// in particle hole-particle particle notation
class chi0_cglattice_phpp : public chi0_phpp {
public:
  // Constructor
  chi0_cglattice_phpp(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                      int nomega4, int size = BlockNum) : chi0_phpp(p, ks, channel, nomega4, size), mu_(p["MU"]) {
    ks_.precompute_dispersion_and_weights(weight_kl_, epsilon_kckl_, epsilon_kckl_plus_pipi_, symmetry_kckl_);
    for (int i = 0; i < num_vert_; ++i) {
      lattice_G_[i].setZero();
      lattice_G_inv_[i].setZero();
    }
  }

  /// get the chi0 out of the Green's function with coarse graining
  void compute_chi0(const fermionic_self_energy &sigma);

  void read(alps::hdf5::archive &ar) { chi0_phpp::read(ar, "chi0_lattice"); };
  void write(alps::hdf5::archive &ar) { chi0_phpp::write(ar, "chi0_lattice"); };

private:
  const double mu_;

  // lattice information
  Matrix<double> epsilon_kckl_;
  Matrix<double> epsilon_kckl_plus_pipi_;
  std::vector<double> weight_kl_;
  // For example, the symmetry factor for dwave is symmetry_kckl_=cos(kx)-cos(ky)
  Matrix<double> symmetry_kckl_;

  std::array<Eigen::Matrix<complex, 2, 2, Eigen::RowMajor>, 2> lattice_G_;
  std::array<Eigen::Matrix<complex, 2, 2, Eigen::RowMajor>, 2> lattice_G_inv_;

  void compute_lattice_G(int omega, int K, int kl, const fermionic_self_energy &sigma);
};

} // namespace nambu
} // namespace SCinverter

#endif //SCINV_CHI0_PHPP_H
