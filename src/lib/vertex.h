//
// Created by Xinyang Dong on 4/27/20.
//

#ifndef SCINV_VERTEX_H
#define SCINV_VERTEX_H

#include <sstream>
#include <complex>
#include <alps/params.hpp>
#include "type.h"
#include "utils.h"
#include "k_space_structure.h"

namespace SCinverter {
namespace nambu {

static constexpr int BlockNum = 16;
static constexpr int BlockInd = 4;

static constexpr size_t VertMatDim = 4; // spin, blockind, fermi1, fermi2
static constexpr size_t VertVecDim = 3; // spin, blockind, fermi1

inline int flatten_ind(int ind1, int ind2) { return ind1 * BlockInd + ind2; }

inline std::pair<int, int> separate_ind(int ind) { return std::pair<int, int> (ind / BlockInd, ind % BlockInd); };

/// \brief The main vertex class.
///
/// This class stores vertices for one Q and one nu
template <size_t N>
class vertex {
public:
  vertex(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
         int nomega4, int size = BlockNum);

  virtual ~vertex() {};

  static constexpr int num_vert_ = 2;

  static constexpr size_t dim = N;

  /// random initialization, mostly for unit tests
  virtual void init_random();

  void set_Qnu(int Q, int nu) { Q_ = Q; nu_ = nu;  }

  /// number of fermionic frequencies
  inline int n_omega4() const { return n_omega4_; }

  /// number of sites
  inline int n_sites() const { return n_sites_; }

  /// Q evaluated
  inline int Q() const { return Q_; }

  /// nu evaluated
  inline int nu() const { return nu_; }

  /// access to channel of this vertex
  inline vert_channel_enum channel() const { return channel_; }

  /// total number of matrix
  inline int num_block() const { return num_block_; }

  /// size of matrix
  inline int mat_size() const { return mat_size_; }

  /// size of matrix
  inline int tot_size() const { return vert_.size(); }

  inline int shift(int spin, int block) const {
    return spin * int (vert_.size() / num_vert_) + block * int (vert_.size() / num_vert_ / num_block_);
  };

  inline const std::vector<int> &block_index() const { return block_index_; }

  /// return fermionic frequency
  inline complex fermi_freq(const int omega) const {
    return complex(0, (2. * omega + 1.) * M_PI / beta_);
  };

  /// fermionic multi-index
  inline int findex(int K, int omega) const { return K * (2 * n_omega4_) + (omega + n_omega4_); }

  /// access function for data
  inline Tensor<complex, N> &vert_tensor() { return vert_; }

  /// const access function for data
  inline const Tensor<complex, N> &vert_tensor() const { return vert_; }

  /// access function for data
  inline Tensor<complex, N> &data() { return vert_; }

  /// const access function for data
  inline const Tensor<complex, N> &data() const { return vert_; }

  /// access function for data
  TensorView<complex, N-2> operator()(int type, int ind1, int ind2) {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      throw std::runtime_error("vert access exceed block number");
    }
    return vert_(type, flatten_ind(ind1, ind2));
  }

  /// const access function for data
  const TensorView<complex, N-2> operator()(int type, int ind1, int ind2) const {
    if (flatten_ind(ind1, ind2) >= num_block_) {
      throw std::runtime_error("vert access exceed block number");
    }
    return vert_(type, flatten_ind(ind1, ind2));
  }

  /// access function for individual element
  complex &operator()(int type, int K, int Kprime, int omega, int omegaprime, int ind1, int ind2);

  /// const access function for individual element
  const complex operator()(int type, int K, int Kprime, int omega, int omegaprime, int ind1, int ind2) const;

  /// read in data from stored file, one (Q, nu), all (K, omega) (K', omegaprime), all blocks
  void read(alps::hdf5::archive &ar, const std::string path_name, int size = num_vert_) {
    for (int n = 0; n < size; ++n) {
      vert_spin_enum s = int2vert_spin_enum(n);
      std::string spin = to_string(s);
      for (int i = 0; i < num_block_; ++i) {
        std::stringstream vertex_name;
        vertex_name << path_name << "_" << to_string(channel_) << "_" << spin
                    << "_Q" << Q_ << "_nu" << nu_ << "_nomega4_" << n_omega4_;
        vertex_name << "_" << int(block_index_[i] / BlockInd) << int(block_index_[i] % BlockInd);

        auto x = vert_(n, i); // a tensor view, need lvalue
        ar[vertex_name.str()] >> x;
      }
    }
  };

  /// write data to stored file, one (Q, nu), all (K, omega) (K', omegaprime), all blocks
  void write(alps::hdf5::archive &ar, const std::string& path_name, int size = num_vert_) const {
    for (int n = 0; n < size; ++n) {
      vert_spin_enum s = int2vert_spin_enum(n);
      std::string spin = to_string(s);
      for (int i = 0; i < num_block_; ++i) {
        std::stringstream vertex_name;
        vertex_name << path_name << "_" << to_string(channel_) << "_" << spin
                    << "_Q" << Q_ << "_nu" << nu_ << "_nomega4_" << n_omega4_;
        vertex_name << "_" << int(block_index_[i] / BlockInd) << int(block_index_[i] % BlockInd);

        ar[vertex_name.str()] << vert_(n, i);
      }
    }
  };

protected:
  const k_space_structure &ks_;
  const vert_channel_enum channel_;
  const std::string file_name_;
  const int num_block_;

  int Q_;
  int nu_;

  const int n_omega4_;
  const int n_sites_;
  const double beta_;
  const double U_;
  const int mat_size_;

  std::vector<int> block_index_;

  // ud and uu
  Tensor<complex, N> vert_;
};

template<>
vertex<VertMatDim>::vertex(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                           int nomega4, const int size);

template<>
vertex<VertVecDim>::vertex(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                           int nomega4, const int size);

template<>
void vertex<VertMatDim>::init_random();

template<>
void vertex<VertVecDim>::init_random();

template<>
complex &vertex<VertMatDim>::operator()(int type, int K, int Kprime, int omega, int omegaprime, int ind1, int ind2);

template<>
complex &vertex<VertVecDim>::operator()(int type, int K, int Kprime, int omega, int omegaprime, int ind1, int ind2);

template<>
const complex vertex<VertMatDim>::operator()(const int type, int K, int Kprime, int omega,
                                             int omegaprime, int ind1, int ind2) const;

template<>
const complex vertex<VertVecDim>::operator()(const int type, int K, int Kprime, int omega,
                                             int omegaprime, int ind1, int ind2) const;

// store several dense matrix
extern template
class vertex<VertMatDim>;

// store sparse diagonal matrix
extern template
class vertex<VertVecDim>;

} // namespace nambu
} // namespace normal

#endif //SCINV_VERTEX_H
