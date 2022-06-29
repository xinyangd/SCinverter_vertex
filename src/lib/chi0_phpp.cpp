
#include "chi0_phpp.h"

namespace SCinverter {
namespace nambu {

void chi0_phpp::get_ud_from_uu(int K, int omega) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  const int ud = vert_spin_enum2int(vert_spin_enum::ud);

  int KpQ = ks_.momenta_sum(K, Q_);
  int kprime = ks_.momenta_diff(ks_.zero_momentum(), KpQ);

  for (int ind = 0; ind < num_block_; ++ind) {
    int i = ind / 4;
    int j = ind % 4;
    if ((i == j) || (i - 2 == j) || (i + 2 == j)) {
      const auto &ref = (*this);
      operator()(ud, K, kprime, omega, -(omega + nu_) - 1, i, j)
        = -ref(uu, K, K, omega, omega, i - (2 * (i % 2)) + 1, j - (2 * (j % 2)) + 1);
    }
    else {
      operator()(ud, K, kprime, omega, -(omega + nu_) - 1, i, j) = -operator()(uu, K, K, omega, omega, i, j);
    }
  }
}

/// \!brief Calculation of the bare cluster susceptibility \f$ \chi_0 \f$.
///
/// For the particle-hole channel see the explicit derivation in Alessandro Eq. 10.
/// the dimension of \f$ \chi \f$ is \f$\frac{1}{\text{energy}^3} \f$.
void chi0_cluster_phpp::compute_chi0(const fermionic_green_function &g2) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  const int ud = vert_spin_enum2int(vert_spin_enum::ud);

  double prefactor = beta_ * n_sites_;

  switch (channel_) {
    case vert_channel_enum::ph: {
      auto cmp_ph = [&](const int K, const int omega){
        int KpQ = ks_.momenta_sum(K, Q_);

        for (int ind = 0; ind < num_block_; ++ind) {
          int i = ind / 4;
          int j = ind % 4;

          int ind1 = i % 2;
          int ind2 = j / 2;
          int ind3 = i / 2;
          int ind4 = j % 2;
          operator()(uu, K, K, omega, omega, i, j)
            = - prefactor * (g2(omega, K, ind1, ind2) * g2(omega + nu_, KpQ, ind3, ind4));
          }
        if (num_block_ == BlockNum)
          get_ud_from_uu(K, omega);
      };

      do_one_fermi_loop(n_omega4_, n_sites_, cmp_ph);
      break;
    }
    case vert_channel_enum::pp: {
      throw std::runtime_error("chi0 pp not implemented yet");
    }
    default:
      throw std::runtime_error("compute cluster chi0 unreachable");
  }
}

/// \!brief calculation of the coarse-grained bare lattice susceptibility.
/// This is done according to Thomas maier's review, equation 127, but note that we do this ONLY for Q
/// vectors which are on cluster vectors. In principle we could generalize this to any q-vector.
///
/// The equation is:
/// \f$ \chi^{0,ph})_{\sigma \sigma'}(K, Kprime, Q, omega, omegaprime, nu)
/// = \delta_{\sigma \sigma'} \delta{KK'} \delta_{\omega\omega'}
/// \frac{N_c}{N} \sum_{\tilde k\in K} G(K+\tilde{k},\omega)G(K+Q+\tilde k, \omega+\nu) \f$
/// the dimension of this quantity is beta^3, same as the dimension of the other chis.
void chi0_cglattice_phpp::compute_chi0(const fermionic_self_energy &sigma) {
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);
  const int ud = vert_spin_enum2int(vert_spin_enum::ud);

  double prefactor = beta_ * n_sites_;

  switch (channel_) {
    case vert_channel_enum::ph: {
      auto cmp_ph = [&](const int K, const int omega) {

        for (int ind = 0; ind < num_block_; ++ind) {
          int i = ind / 4;
          int j = ind % 4;
          operator()(uu, K, K, omega, omega, i, j) = 0.;

          int ind1 = i % 2;
          int ind2 = j / 2;
          int ind3 = i / 2;
          int ind4 = j % 2;

          double weight_sum = 0.;
          for (int kl = 0; kl < weight_kl_.size(); ++kl) {

            compute_lattice_G(omega, K, kl, sigma);

            operator()(uu, K, K, omega, omega, i, j) +=
              weight_kl_[kl] * lattice_G_[0](ind1, ind2) * lattice_G_[1](ind3, ind4);

            weight_sum += weight_kl_[kl];
          }
          if (std::abs(weight_sum - 1) > 1.e-12)
            throw std::logic_error("problem with the weight sum");

          operator()(uu, K, K, omega, omega, i, j) *= -prefactor;

        } // end ind

        get_ud_from_uu(K, omega);
      };

      do_one_fermi_loop(n_omega4_, n_sites_, cmp_ph);
      break;
    }

    case vert_channel_enum::pp: {
      throw std::runtime_error("chi0 pp not implemented yet");
    }

    default:
      throw std::runtime_error("compute lattice chi0 unreachable");
  }
}

void chi0_cglattice_phpp::compute_lattice_G(int omega, int K, int kl, const fermionic_self_energy &sigma) {
  int KpQ = ks_.momenta_sum(K, Q_);

  lattice_G_inv_[0](0, 0) = fermi_freq(omega) + mu_ - epsilon_kckl_(K, kl) - sigma(omega, K, 0, 0);
  lattice_G_inv_[0](0, 1) = - sigma(omega, K, 0, 1);
  lattice_G_inv_[0](1, 0) = - sigma(omega, K, 1, 0);
  lattice_G_inv_[0](1, 1) = fermi_freq(omega) - mu_ + epsilon_kckl_(K, kl) - sigma(omega, K, 1, 1);

  lattice_G_inv_[1](0, 0) = fermi_freq(omega + nu_) + mu_ - epsilon_kckl_(KpQ, kl) - sigma(omega + nu_, KpQ, 0, 0);
  lattice_G_inv_[1](0, 1) = - sigma(omega + nu_, KpQ, 0, 1);
  lattice_G_inv_[1](1, 0) = - sigma(omega + nu_, KpQ, 1, 0);
  lattice_G_inv_[1](1, 1) = fermi_freq(omega + nu_) - mu_ + epsilon_kckl_(KpQ, kl) - sigma(omega + nu_, KpQ, 1, 1);

  for (int i = 0; i < 2; ++i) {
    lattice_G_[i] = lattice_G_inv_[i].inverse();
  }
}

} // namespace nambu
} // namespace SCinverter