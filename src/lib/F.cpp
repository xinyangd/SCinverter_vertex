
#include "F.h"

namespace SCinverter {
namespace nambu {

constexpr std::array<int, BlockNum> F_phpp::ind_map;
constexpr std::array<int, BlockNum> F_phpp::rev_ind_map;

void F_phpp::invert_F_phpp(alps::hdf5::archive &ar, chi_cluster_phpp &chi, chi0_cluster_phpp &chi0) {

  if ((chi.Q() != chi0.Q()) || (chi.Q() != Q())) {
    throw std::runtime_error("invert F momentum mismatch");
  }
  if ((chi.nu() != chi0.nu()) || (chi.nu() != nu())) {
    throw std::runtime_error("invert F frequency mismatch");
  }

  if (num_block_ == 1) {
    if ((chi.num_block() != 1 || chi0.num_block() != 1))
      throw std::runtime_error("chi and chi0 must have same size as F");

    chi.read(ar);
    chi0.read(ar);

    compute_F(chi, chi0);
  }
  else {
    if ((chi.num_block() != BlockNum || chi0.num_block() != BlockNum))
      throw std::runtime_error("chi and chi0 must in separate block form");

    read_block(ar, "chi_cluster", chi.vert_tensor());
    read_block(ar, "chi0_cluster", chi0.vert_tensor());

    chi_minus_chi0(chi, chi0); // chi - chi0
    inverse_chi0(chi0); // give value to chi0_inv;
    compute_F(chi);
  }
}

void F_phpp::invert_F_phpp(chi_cluster_phpp &chi, chi0_cluster_phpp &chi0) {

  if ((chi.Q() != chi0.Q()) || (chi.Q() != Q())) {
    throw std::runtime_error("invert F momentum mismatch");
  }
  if ((chi.nu() != chi0.nu()) || (chi.nu() != nu())) {
    throw std::runtime_error("invert F frequency mismatch");
  }

  if (num_block_ == 1) {
    if ((chi.num_block() != 1 || chi0.num_block() != 1))
      throw std::runtime_error("chi and chi0 must have same size as F");

    compute_F(chi, chi0);
  }
  else {
    if ((chi.num_block() != BlockNum || chi0.num_block() != BlockNum))
      throw std::runtime_error("chi and chi0 must in separate block form");

    shuffle_blocks(chi.vert_tensor());
    shuffle_blocks(chi0.vert_tensor());

    chi_minus_chi0(chi, chi0); // chi - chi0
    inverse_chi0(chi0); // give value to chi0_inv;
    compute_F(chi);
  }
}

void F_phpp::check_inverse_chi0_matrix(const chi_cluster_phpp &chi, const chi0_cluster_phpp &chi0) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);

  Matrix<complex> chi0_mat_uu(4 * chi0.mat_size(), 4 * chi0.mat_size());
  Matrix<complex> chi0_inv_uu(4 * chi0.mat_size(), 4 * chi0.mat_size());
  Matrix<complex> chi0_mat_dd(4 * chi0.mat_size(), 4 * chi0.mat_size());
  Matrix<complex> chi0_inv_dd(4 * chi0.mat_size(), 4 * chi0.mat_size());

  for (int i = 0; i < BlockInd; ++i) {
    for (int j = 0; j < BlockInd; ++j) {
      chi0_mat_uu.block(i * chi0.mat_size(), j * chi0.mat_size(), chi0.mat_size(), chi0.mat_size())
      = Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), chi0.shift(uu, flatten_ind(i, j))).asDiagonal();

      int spin_sign = (i == j || BlockInd - 1 - i == j) ? 1 : -1;
      chi0_mat_dd.block(i * chi0.mat_size(), j * chi0.mat_size(), chi0.mat_size(), chi0.mat_size())
      = spin_sign * Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), chi0.shift(uu, flatten_ind(i, j))).asDiagonal();
    }
  }
  chi0_inv_uu = chi0_mat_uu.inverse();
  chi0_inv_dd = chi0_mat_dd.inverse();

  Matrix<complex> diag(chi0.mat_size(), chi0.mat_size());
  int x1, x2;
  for (int i = 0; i < BlockNum; ++i) {
    std::cout << "i " << i << std::endl;
    std::tie<int, int> (x1, x2) = separate_ind(i);

    diag.setZero();
    auto m = chi0_inv_uu.block(x1 * chi0.mat_size(), x2 * chi0.mat_size(), chi0.mat_size(), chi0.mat_size());
    diag.diagonal() = m.diagonal();
    auto res = m - diag;

    auto maxdiff = (m.diagonal().transpose() - chi0_inv_[i]).cwiseAbs().maxCoeff();
    std::cout << maxdiff << std::endl;
    auto maxres = res.cwiseAbs().maxCoeff();
    std::cout << maxres << std::endl;

    auto n = chi0_inv_dd.block(x1 * chi0.mat_size(), x2 * chi0.mat_size(), chi0.mat_size(), chi0.mat_size());
    int spin_sign = (x1 == x2 || BlockInd - 1 - x1 == x2) ? 1 : -1;
    auto spin_diff = m - spin_sign * n;
    auto max_spin_diff = spin_diff.cwiseAbs().maxCoeff();
    std::cout << max_spin_diff << std::endl;
  }
}

void F_phpp::chi_minus_chi0(chi_cluster_phpp &chi, const chi0_cluster_phpp &chi0) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  const int ud = vert_spin_enum2int(vert_spin_enum::ud);

  for (int i = 0; i < BlockNum; ++i) {
    int shift = chi0.shift(uu, i);
    chi.vert_tensor()(uu, i).matrix().diagonal() -= Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shift);

    for (int omega = -n_omega4_; omega < n_omega4_; ++omega) {
      for (int K = 0; K < n_sites_; ++K) {
        int KpQ = ks_.momenta_sum(K, Q_);
        int kprime = ks_.momenta_diff(ks_.zero_momentum(), KpQ);

        int ind1, ind2;
        std::tie<int, int>(ind1, ind2) = separate_ind(i);

        chi(ud, K, kprime, omega, -(omega + nu_) - 1, ind1, ind2)
          -= chi0(ud, K, kprime, omega, -(omega + nu_) - 1, ind1, ind2);
      }
    }
  }
}

void F_phpp::inverse_chi0(const chi0_cluster_phpp &chi0) {

  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  const int len = chi0.mat_size();

  // lower right block D^-1
  std::array<int, BlockInd> D_ind = {10, 11, 14, 15};
  std::array<int, BlockInd> D_shift;
  for (int i = 0; i < BlockInd; ++i) {
    D_shift[i] = chi0.shift(uu, D_ind[i]);
    core_[i] = Tensor_VecView(chi0.vert_tensor(), len, D_shift[i]);
  }
  compute_2by2_inv(core_inv1_, core_);

  // the other inverse (A - B D^-1 C)^-1
  for (int i = 0; i < BlockInd; ++i) {
    int r = i / 2;
    int c = i % 2;

    // chi0.vert_tensor()(uu, flatten_ind(r, c));
    int shiftA = chi0.shift(uu, flatten_ind(r, c));
    core_[i].array() = Tensor_VecView(chi0.vert_tensor(), len, shiftA).array();

    for (int x = 0; x < 2; ++x) {
      for (int y = 0; y < 2; ++y) {
        int shiftB = chi0.shift(uu, flatten_ind(r, 2 + x));
        int shiftC = chi0.shift(uu, flatten_ind(2 + y, c));

        core_[i].array() -=
          Tensor_VecView(chi0.vert_tensor(), len, shiftB).array() *
          core_inv1_[2 * x + y].array() *
          Tensor_VecView(chi0.vert_tensor(), len, shiftC).array();
      }
    }
  }
  compute_2by2_inv(core_inv2_, core_);

  // final inverse
  for (int i = 0; i < chi0_inv_.size(); ++i) {
    chi0_inv_[i].setZero();

    int inv1, inv2;
    std::tie<int, int>(inv1, inv2) = separate_ind(i);

    // upper left
    if ((int(inv1 / 2) + int(inv2 / 2)) == 0) {
      chi0_inv_[i] = core_inv2_[inv1 * 2 + inv2];
    }
    // upper right
    else if ((int(inv1 / 2) == 0) && (int(inv2 / 2) == 1)) {
      for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
          int shiftUR = chi0.shift(uu, flatten_ind(x, y + 2));
          chi0_inv_[i].array() -= core_inv2_[inv1 * 2 + x].array() *
                                  Tensor_VecView(chi0.vert_tensor(), len, shiftUR).array() *
                                  core_inv1_[y * 2 + (inv2 - 2)].array();
        }
      }
    }
    // lower left
    else if ((int(inv1 / 2) == 1) && (int(inv2 / 2) == 0)) {
      for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
          int shiftLL = chi0.shift(uu, flatten_ind(x + 2, y));
          chi0_inv_[i].array() -= core_inv1_[(inv1 - 2) * 2 + x].array() *
                                  Tensor_VecView(chi0.vert_tensor(), len, shiftLL).array() *
                                  core_inv2_[y * 2 + inv2].array();
        }
      }
    }
    // lower right
    else {
      chi0_inv_[i].array() = core_inv1_[(inv1 - 2) * 2 + (inv2 - 2)].array();
      for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
          int shiftLR = chi0.shift(uu, flatten_ind(x + 2, y));
          chi0_inv_[i].array() -= chi0_inv_[flatten_ind(y, inv2)].array() *
                                  Tensor_VecView(chi0.vert_tensor(), len, shiftLR).array() *
                                  core_inv1_[(inv1 - 2) * 2 + x].array();
        }
      }
    }
  }
}

// this function only works because for 2*2 blockwise diagonal matrix, block1 * block2 = block2 * block1
void F_phpp::compute_2by2_inv(std::array<Vector<complex>, BlockInd> &core_inv,
                              const std::array<Vector<complex>, BlockInd> &core) {

  core_inv[0] = core[3];
  core_inv[1] = -core[1];
  core_inv[2] = -core[2];
  core_inv[3] = core[0];

  chi0_det_inv_.array()
    = 1. / (core_inv[0].array() * core_inv[3].array()- core_inv[1].array() * core_inv[2].array());

  for (int i = 0; i < BlockInd; ++i) {
    core_inv[i].array() *= chi0_det_inv_.array();
  }
}

void F_phpp::compute_F(const chi_cluster_phpp &chi) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  double prefactor = - beta_ * beta_ * n_sites_ * n_sites_;
  vert_.set_zero();

  // two spin
  for (int s = 0; s < num_vert_; ++s) {
    for (int i = 0; i < BlockNum; ++i) {
      int f1, f2;
      std::tie<int, int>(f1, f2) = separate_ind(i);

      for (int j = 0; j < BlockInd; ++j) {
        for (int k = 0; k < BlockInd; ++k) {

          // fix sign for spin dd
          int spin_sign = (s == uu || (k == f2 || BlockInd - 1 - k == f2)) ? 1 : -1;

          int shiftChi = chi.shift(s, flatten_ind(j, k));
          vert_(s, i).matrix() +=
            chi0_inv_[flatten_ind(f1, j)].asDiagonal() *
            Tensor_MatView(chi.vert_tensor(), chi.mat_size(), chi.mat_size(), shiftChi) *
            chi0_inv_[flatten_ind(k, f2)].asDiagonal() * spin_sign;
        }
      }
    }
  }
  vert_ *= prefactor;
}

void F_phpp::compute_F(chi_cluster_phpp &chi, const chi0_cluster_phpp &chi0) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);

  int shift = chi0.shift(uu, 0);
  chi.vert_tensor()(uu, 0).matrix().diagonal() -= Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shift);

  double prefactor = - beta_ * beta_ * n_sites_ * n_sites_;
  vert_.set_zero();

  // two spin
  for (int s = 0; s < num_vert_; ++s) {
    vert_(s, 0).matrix() = Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shift).cwiseInverse().asDiagonal() *
                           chi.vert_tensor()(s, 0).matrix() *
                           Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shift).cwiseInverse().asDiagonal();
  }
  vert_ *= prefactor;
}

void F_phpp::compute_chi_from_F(alps::hdf5::archive &ar, chi_phpp &chi, chi0_phpp &chi0) {

  if ((chi.num_block() != num_block_ || chi0.num_block() != num_block_))
    throw std::runtime_error("chi and chi0 must have same size as F");

  if ((chi.Q() != chi0.Q()) || (chi.Q() != Q())) {
    throw std::runtime_error("invert F momentum mismatch");
    if ((chi.nu() != chi0.nu()) || (chi.nu() != nu())) {
      throw std::runtime_error("invert F frequency mismatch");
    }
  }

  read_block(ar, "F_cluster", vert_tensor());
  read_block(ar, "chi0_cluster", chi0.vert_tensor());

  compute_mchi0Fchi0(chi, chi0);
  add_chi0(chi, chi0);
}

void F_phpp::compute_mchi0Fchi0(chi_phpp &chi, const chi0_phpp &chi0) const {
  double prefactor = - beta_ * beta_ * n_sites_ * n_sites_;
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);

  chi.vert_tensor().set_zero();
  // two spin
  for (int s = 0; s < num_vert_; ++s) {
    for (int i = 0; i < num_block_; ++i) {
      int f1, f2;
      std::tie<int, int>(f1, f2) = separate_ind(i);
      for (int j = 0; j < int(std::sqrt(num_block_)); ++j) {
        for (int k = 0; k < int(std::sqrt(num_block_)); ++k) {

          // chi0.vert_tensor()(uu, flatten_ind(f1, j));
          int shiftA = chi0.shift(uu, flatten_ind(f1, j));
          // chi0.vert_tensor()(uu, flatten_ind(k, f2));
          int shiftB = chi0.shift(uu, flatten_ind(k, f2));
          // vert_(s, flatten_ind(j, k)).matrix()
          int shiftF = shift(s, flatten_ind(j, k));

          // fix sign for spin dd
          int spin_sign = (s == uu || (k == f2 || BlockInd - 1 - k == f2)) ? 1 : -1;

          chi.vert_tensor()(s, i).matrix() +=
            Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shiftA).asDiagonal() *
            Tensor_MatView(vert_, mat_size_, mat_size_, shiftF).matrix() *
            Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shiftB).asDiagonal() * spin_sign;
        }
      }
    }
  }
  chi.vert_tensor() /= prefactor;
}

void F_phpp::add_chi0(chi_phpp &chi, const chi0_phpp &chi0) const {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  const int ud = vert_spin_enum2int(vert_spin_enum::ud);

  for (int i = 0; i < num_block_; ++i) {
    //  chi.vert_vec(uu)[i].diagonal() += chi0.vert_vec(uu)[i];
    int shift = chi0.shift(uu, i);
    chi.vert_tensor()(uu, i).matrix().diagonal() += Tensor_VecView(chi0.vert_tensor(), chi0.mat_size(), shift);

    for (int omega = -n_omega4_; omega < n_omega4_; ++omega) {
      for (int K = 0; K < n_sites_; ++K) {
        int KpQ = ks_.momenta_sum(K, Q_);
        int kprime = ks_.momenta_diff(ks_.zero_momentum(), KpQ);

        int ind1, ind2;
        std::tie<int, int>(ind1, ind2) = separate_ind(i);

        chi(ud, K, kprime, omega, -(omega + nu_) - 1, ind1, ind2)
          += chi0(ud, K, kprime, omega, -(omega + nu_) - 1, ind1, ind2);
      }
    }
  }
}

} // namespace nambu
} // namespace SCinverter