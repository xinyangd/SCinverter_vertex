
#include <iostream>
#include <sstream>
#include "mc_vertex.h"
#include "utils.h"

namespace SCinverter {
namespace nambu {

void mc_vertex::rearrange(alps::hdf5::archive &temp, alps::hdf5::archive &ar, bool print) {

  std::string channel = to_string(channel_);
  std::string e_or_v = to_string(e_or_v_);
  std::string base_name = "G4_" + channel + "_";

  // buff for one Q, all nu, all (K, omega) (Kprime, omegaprime)
  Tensor<std::complex<double>, 3> buf(2 * n_omega4_bose_ + 1, 2 * n_sites_ * n_omega4_, 2 * n_sites_ * n_omega4_);
  bool print_read = print;
  for (int s = 0; s < 2; ++s) {
    vert_spin_enum spin = int2vert_spin_enum(s);
    for (int i = 0; i < block_index_.size(); ++i) {
      for (int Q = 0; Q < n_sites_; ++Q) {

        for (int K = 0; K < n_sites_; ++ K) {
          for (int Kprime = 0; Kprime < n_sites_; ++Kprime) {
            int symm_K = ks_.symmetrized_K_index(K, Kprime, Q);
            int symm_Kprime = ks_.symmetrized_Kprime_index(K, Kprime, Q);
            int symm_Q = ks_.symmetrized_Q_index(K, Kprime, Q);

            // one Q, K, Kprime, all frequencies
            read(ar, symm_Q, symm_K, symm_Kprime, block_index_[i], s, print_read);

            for (int nu = -n_omega4_bose_; nu <= n_omega4_bose_; ++nu) {
              for (int omega = -n_omega4_; omega < n_omega4_; ++omega) {
                for (int omegaprime = -n_omega4_; omegaprime < n_omega4_; ++omegaprime) {
                  buf(nu + n_omega4_bose_, findex(K, omega), findex(Kprime, omegaprime))
                    = vertex_[freqindex3(nu, omega, omegaprime)];
                } // loop omegaprime
              } // loop omega
            } // loop nu
          } // loop Kprime
        } // loop K

        // write result out
        for (int nu = -n_omega4_bose_; nu <= n_omega4_bose_; ++nu) {
          std::stringstream vertex_name;
          vertex_name
            << "/simulation/results/" << base_name << to_string(spin) << "_Q" << Q << "_nu" << nu
            << "_" << int(block_index_[i] / 4) << int(block_index_[i] % 4) << "/mean/" << e_or_v;
          temp[vertex_name.str()] << buf(nu + n_omega4_bose_);
        }

      } // loop Q
    } // loop ind
  } // loop spin

  if (print) {
    std::cout << "mc vertex rearrange done" << std::endl;
  }
}

void mc_vertex::read(alps::hdf5::archive &ar, int Q, int K, int Kprime, int ind, int s, bool print) {
  std::string channel = to_string(channel_);
  std::string e_or_v = to_string(e_or_v_);
  vert_spin_enum spin = int2vert_spin_enum(s);

  if (print)
    std::cout << "reading sign from sim.h5 file" << std::endl;
  double sign_result;
  ar >> alps::make_pvp("/simulation/results/Sign/mean/value", sign_result);

  if (print) {
    std::cout << "Using value of Sign = " << sign_result << std::endl;
    std::cout << "tot_freq_size " << tot_freq_size_ << std::endl;
  }

  std::string base_name = "G4_Q_K_Kprime_nu_omega_omegaprime_" + channel + "_";
  std::stringstream vertex_re_name, vertex_im_name;
  vertex_re_name
    << "/simulation/results/" << base_name << to_string(spin) << "_re_" << Q << "_" << K << "_" << Kprime
    << "_" << int(ind / 4) << int(ind % 4) << "_times_sign/mean/" << e_or_v;
  vertex_im_name
    << "/simulation/results/" << base_name << to_string(spin) << "_im_" << Q << "_" << K << "_" << Kprime
    << "_" << int(ind / 4) << int(ind % 4) << "_times_sign/mean/" << e_or_v;

  const int dx = 1, dy = 2;
  int start_ind = 0;
  double *real_location, *imag_location;
  real_location = (double *) &(vertex_[start_ind]);
  imag_location = real_location + 1;

  ar[vertex_re_name.str()] >> res_buffer_;
  blas::dcopy_(&tot_freq_size_, &(res_buffer_[0]), &dx, real_location, &dy);
  ar[vertex_im_name.str()] >> res_buffer_;
  blas::dcopy_(&tot_freq_size_, &(res_buffer_[0]), &dx, imag_location, &dy);

  std::for_each(vertex_.begin(), vertex_.end(), [&](std::complex<double> &n){ n /= sign_result; });
}

} // namespace nambu
} // namespace SCinverter