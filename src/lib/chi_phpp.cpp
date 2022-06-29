
#include "chi_phpp.h"

namespace SCinverter {
namespace nambu {

void chi_cluster_phpp::compute_chi(alps::hdf5::archive &ar_in, const fermionic_green_function &g2) {
  double prefactor = beta_ * n_sites_;

  int s, i, j, ind1, ind2, ind3, ind4, temp_sign;
  auto subs_discon = [&](const int K, const int Kprime, const int omega, const int omegaprime){
    operator()(s, K, Kprime, omega, omegaprime, i, j) -=
      double(temp_sign) * g2(omega, K, ind1, ind2) * g2(omegaprime, Kprime, ind3, ind4) * prefactor;
  };

  switch (channel_) {
    case vert_channel_enum::ph: {

      std::string channel = to_string(vert_channel_enum::ph);
      std::string base_name = "G4_" + channel + "_";

      for (s = 0; s < num_vert_; ++s) {
        vert_spin_enum spin = int2vert_spin_enum(s);

        for (int ind = 0; ind < num_block_; ++ind) {
          i = ind / 4;
          j = ind % 4;

          std::stringstream vertex_name;
          vertex_name
            << "/simulation/results/" << base_name << to_string(spin) << "_Q" << Q_ << "_nu" << nu_
            << "_" << i << j << "/mean/value";
          auto a = vert_(s, ind);
          ar_in[vertex_name.str()] >> a;

          // subtract disconnected part
          if (nu_ == 0 && Q_ == ks_.zero_momentum()) {
            temp_sign =
              (spin == vert_spin_enum::uu || // uu is always 1
               (i == j || (i - 2) == j || (i + 2) == j)) ? 1 : -1;

            std::cout << "spin: " << s << std::endl;
            std::cout << "index: " << i << j << std::endl;
            std::cout << "disconnect sign: " << temp_sign << std::endl;

            ind1 = i / 2;
            ind2 = j / 2;
            ind3 = i % 2;
            ind4 = j % 2;

            do_two_fermi_loops(n_omega4_, n_sites_, subs_discon);
          }
        }
      }
      break;
    }
    case vert_channel_enum::pp: {
      throw std::runtime_error("chi cluster pp not implemented");
    }
    default:
      throw std::runtime_error("compute chi cluster unreachable");
  }
}

} // namespace nambu
} // namespace SCinverter