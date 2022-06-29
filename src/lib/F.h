
#ifndef SCINV_F_H
#define SCINV_F_H

#include "vertex.h"
#include "chi_phpp.h"
#include "chi0_phpp.h"

namespace SCinverter {
namespace nambu {

/// Class for all full vertex
class F : public vertex<VertMatDim> {
public:
  // Constructor
  F(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
      int size = BlockNum) : vertex<VertMatDim>(p, ks, channel, p["NOMEGA4"], size) {};

  virtual ~F() {};
};

/// Class for the F (the 'complete' vertex function) in particle hole-particle particle notation
class F_phpp : public F {
public:
  // Constructor
  F_phpp(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
         int size = BlockNum) : F(p, ks, channel, size), chi0_det_inv_(mat_size_) {
    for (int i = 0; i < BlockNum; ++i) {
      chi0_inv_[i].resize(mat_size_);
    }
    for (int i = 0; i < BlockInd; ++i) {
      core_[i].resize(mat_size_);
      core_inv1_[i].resize(mat_size_);
      core_inv2_[i].resize(mat_size_);
    }
  };

  /// compute the \f$F\f$ vertex out of \f$chi\f$ and \f$chi_0\f$ with eigen matrix inverse
  void invert_F_phpp(alps::hdf5::archive &ar, chi_cluster_phpp &chi, chi0_cluster_phpp &chi0);
  void invert_F_phpp(chi_cluster_phpp &chi, chi0_cluster_phpp &chi0);

  void read(alps::hdf5::archive &ar) { F::read(ar, "F_cluster"); };
  void write(alps::hdf5::archive &ar) { F::write(ar, "F_cluster"); };

  /// read in blocks of chi and chi0 and reorder
  template<size_t N>
  void read_block(alps::hdf5::archive &ar, const std::string& path_name, Tensor<complex, N> &vert) const {

    for (int n = 0; n < vert.shape()[0]; ++n) {
      vert_spin_enum s = int2vert_spin_enum(n);
      std::string spin = to_string(s);
      for (int i = 0; i < vert.shape()[1]; ++i) {
        std::stringstream vertex_name;
        vertex_name << path_name << "_" << to_string(channel_) << "_" << spin << "_Q" << Q_
                    << "_nu" << nu_ << "_nomega4_" << n_omega4_ << "_" << int(i / BlockInd) << int(i % BlockInd);
        auto a = vert(n, rev_ind_map[i]);
        ar[vertex_name.str()] >> a;
      }
    }
  }

  /// reorder and write blocks of F
  template<size_t N>
  void write_block(alps::hdf5::archive &ar, const std::string& path_name, Tensor<complex, N> &vert) const {

    for (int n = 0; n < vert.shape()[0]; ++n) {
      vert_spin_enum s = int2vert_spin_enum(n);
      std::string spin = to_string(s);
      for (int i = 0; i < vert.shape()[1]; ++i) {
        std::stringstream vertex_name;
        vertex_name << path_name << "_" << to_string(channel_) << "_" << spin << "_Q" << Q_
                    << "_nu" << nu_ << "_nomega4_" << n_omega4_ << "_" << int(i / BlockInd) << int(i % BlockInd);
        ar[vertex_name.str()] << vert(n, rev_ind_map[i]);
      }
    }
  }

  /// recover chi from F and chi0, mainly for debugging purpose
  void compute_chi_from_F(alps::hdf5::archive &ar, chi_phpp &chi, chi0_phpp &chi0);

  // brutal force inverse check used in test
  void check_inverse_chi0_matrix(const chi_cluster_phpp &chi, const chi0_cluster_phpp &chi0);

private:
  // map from measurement to matrix multiplication
  static constexpr std::array<int, BlockNum> ind_map = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};

  // map from matrix multiplication to measurement
  static constexpr std::array<int, BlockNum> rev_ind_map = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};

  std::array<Vector<complex>, BlockNum> chi0_inv_;

  Vector<complex> chi0_det_inv_;
  std::array<Vector<complex>, BlockInd> core_;
  std::array<Vector<complex>, BlockInd> core_inv1_;
  std::array<Vector<complex>, BlockInd> core_inv2_;

  // reorder blocks for invert
  template<size_t N>
  void shuffle_blocks(Tensor<complex, N> &vert) {
    Tensor<complex, N> temp(vert);
    for (int n = 0; n < vert.shape()[0]; ++n) {
      for (int i = 0; i < vert.shape()[1]; ++i) {
        vert(n, rev_ind_map[i]) = temp(n, i);
      }
    }
  }

  // compute chi - chi0
  void chi_minus_chi0(chi_cluster_phpp &chi, const chi0_cluster_phpp &chi0);
  // inverse
  void inverse_chi0(const chi0_cluster_phpp &chi0);

  void compute_2by2_inv(std::array<Vector<complex>, BlockInd> &core_inv,
                        const std::array<Vector<complex>, BlockInd> &core);

  void compute_F(const chi_cluster_phpp &chi);

  // compute F in normal state, with only one block
  void compute_F(chi_cluster_phpp &chi, const chi0_cluster_phpp &chi0);

  /*
   * Below are functions used in cmp_chi_from_F
   */

  // compute -chi_0 F chi_0 -> chi
  void compute_mchi0Fchi0(chi_phpp &chi, const chi0_phpp &chi0) const;

  // compute chi0 + (-chi_0 F chi_0) -> chi
  void add_chi0(chi_phpp &chi, const chi0_phpp &chi0) const;

  /*
   * End
   */
};

} // namespace nambu
} // namespace SCinverter

#endif //SCINV_F_H
