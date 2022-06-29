
#include "vertex.h"

namespace SCinverter {
namespace nambu {

template<>
vertex<VertMatDim>::vertex(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                           int nomega4, int size): ks_(ks), channel_(channel),
                                                   file_name_(p["OUT_FILE_NAME"].as<std::string>()),
                                                   num_block_(size), n_omega4_(nomega4), n_sites_(p["dca.SITES"]),
                                                   beta_(p["BETA"]), U_(p["U"]),
                                                   mat_size_(2 * n_sites_ * n_omega4_) {

  vert_.reshape(num_vert_, num_block_, mat_size_, mat_size_);
  vert_.set_zero();

  block_index_.resize(num_block_);
  for (int i = 0; i < num_block_; ++i) {
    block_index_[i] = i;
  }
}

template<>
vertex<VertVecDim>::vertex(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel,
                           int nomega4, int size): ks_(ks), channel_(channel),
                                                   file_name_(p["OUT_FILE_NAME"].as<std::string>()),
                                                   num_block_(size), n_omega4_(nomega4), n_sites_(p["dca.SITES"]),
                                                   beta_(p["BETA"]), U_(p["U"]),
                                                   mat_size_(2 * n_sites_ * n_omega4_) {
  vert_.reshape(num_vert_, num_block_, mat_size_);
  vert_.set_zero();

  block_index_.resize(num_block_);
  for (int i = 0; i < num_block_; ++i) {
    block_index_[i] = i;
  }
}

template<>
void vertex<VertMatDim>::init_random() {
  for (int i = 0; i < num_vert_; ++i) {
    for (int j = 0; j < num_block_; ++j) {
      vert_(i, j).matrix() = Matrix<complex>::Random(mat_size_, mat_size_);
    }
  }
}

template<>
void vertex<VertVecDim>::init_random() {
  for (int i = 0; i < num_block_; ++i) {
    auto a = vert_(vert_spin_enum2int(vert_spin_enum::uu), i);
    Tensor_VecView(a, mat_size_) = Vector<complex>::Random(mat_size_);
  }
}

template<>
complex &vertex<VertMatDim>::operator()(int type, int K, int Kprime, int omega, int omegaprime, int ind1, int ind2) {
  if (flatten_ind(ind1, ind2) >= num_block_) {
    throw std::runtime_error("vert element exceed block number");
  }
  return vert_(type, flatten_ind(ind1, ind2), findex(K, omega), findex(Kprime, omegaprime));
}

template<>
complex &vertex<VertVecDim>::operator()(int type, int K, int Kprime,
                                        int omega, int omegaprime, int ind1, int ind2) {
  if (flatten_ind(ind1, ind2) >= num_block_) {
    throw std::runtime_error("vert element exceed block number");
  }
  switch (type) {
    //ud
    case 0: {
      int KpQ = ks_.momenta_sum(K, Q_);
      int mKprime = ks_.momenta_diff(ks_.zero_momentum(), Kprime);

      if ((KpQ != mKprime) || (omega + nu_ != -omegaprime - 1))
        throw std::runtime_error("Eigen::Vector vert ele unreachable");

      return vert_(type, flatten_ind(ind1, ind2), findex(K, omega));
    }
    //uu, diagonal
    case 1: {
      if ((K != Kprime) || (omega != omegaprime))
        throw std::runtime_error("Eigen::Vector vert ele unreachable");

      return vert_(type, flatten_ind(ind1, ind2), findex(K, omega));
    }
    default :
      throw std::runtime_error("vertex element type unreachable");
  }
}

template<>
const complex vertex<VertMatDim>::operator()(int type, int K, int Kprime,
                                             int omega, int omegaprime, int ind1, int ind2) const {
  if (flatten_ind(ind1, ind2) < num_block_) {
    return vert_(type, flatten_ind(ind1, ind2), findex(K, omega), findex(Kprime, omegaprime));
  }
  return complex(0, 0);
}

template<>
const complex vertex<VertVecDim>::operator()(int type, int K, int Kprime, int omega,
                                             int omegaprime, int ind1, int ind2) const {
  if (flatten_ind(ind1, ind2) >= num_block_) {
    return complex(0, 0);
  }
  switch (type) {
    //ud
    case 0: {
      int KpQ = ks_.momenta_sum(K, Q_);
      int mKprime = ks_.momenta_diff(ks_.zero_momentum(), Kprime);
      if ((KpQ == mKprime) && (omega + nu_ == -omegaprime - 1)) {
        return vert_(type, flatten_ind(ind1, ind2), findex(K, omega));
      }
      return complex(0, 0);
    }
    //uu, diagonal
    case 1: {
      if ((K == Kprime) && (omega == omegaprime)) {
        return vert_(type, flatten_ind(ind1, ind2), findex(K, omega));
      }
      return complex(0, 0);
    }
    default :
      throw std::runtime_error("vertex element type unreachable");
  }
}

// store several dense matrix
template class vertex<VertMatDim>;

// store sparse diagonal matrix
template class vertex<VertVecDim>;

} // namespace nambu
} // namespace SCinverter