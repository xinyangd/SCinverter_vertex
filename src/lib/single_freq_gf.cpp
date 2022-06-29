
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <alps/hdf5.hpp>
#include "single_freq_gf.h"

namespace SCinverter {
namespace nambu {

void fermionic_matsubara_function::write_single_freq(const std::string &out_file_name_gf, bool hr) const {
  int omega_start = hr ? 0 : -nt_;

  std::ofstream outfile_gf(out_file_name_gf.c_str());
  outfile_gf << std::setprecision(15);
  for (int omega = omega_start; omega < nt_; ++omega) {
    outfile_gf << (2 * (omega) + 1) * M_PI / beta_;
    for (int K = 0; K < ns_; ++K) {
      for (int i = 0; i < nf_; ++i) {
        for (int j = 0; j < nf_; ++j) {
          outfile_gf << " " << operator()(omega, K, i, j).real() << " " << operator()(omega, K, i, j).imag();
        }
      }
    }
    outfile_gf << std::endl;
  }
}

void fermionic_matsubara_function::read_single_freq_ff(const std::string &in_file_name_gf, bool print) {
  std::ifstream infile_gf(in_file_name_gf.c_str());
  if (print)
    std::cout << "read single freq from file" << in_file_name_gf << std::endl;
  if (!infile_gf.is_open())
    throw std::runtime_error("tried to read file that could not be opened: " + in_file_name_gf);

  double ignored;
  std::string ignored_line;

  // There are four lines of grid information
  char c;
  infile_gf >> c;
  if (c == '#') {
    for (int i = 0; i < 4; ++i)
      std::getline(infile_gf, ignored_line);
  }

  double gin_real, gin_imag;
  for (int omega = 0; omega < nt_; ++omega) {
    infile_gf >> ignored;
    for (int K = 0; K < ns_; ++K) {
      for (int i = 0; i < nf_; ++i) {
        for (int j = 0; j < nf_; ++j) {
          infile_gf >> gin_real >> gin_imag;
          operator()(omega, K, i, j) = std::complex<double>(gin_real, gin_imag);
        }
      }
    }
  }
}

std::complex<double> fermionic_matsubara_function::hifreq(int n, int site, int f1, int f2) const {
  double wn = (2 * n + 1) * M_PI / beta_;
  std::complex<double> iwn(0., wn);

  return c1(site, f1, f2) / iwn + c2(site, f1, f2) / (iwn * iwn) + c3(site, f1, f2) / (iwn * iwn * iwn);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void fermionic_green_function::set_hubbard_hifreq_tails(double U) {
  for (int i = 0; i < ns_; i++) {
    for (int f = 0; f < nf_; f++) {
      c1_[i * nf_ * nf_ + f * nf_ + f] = 1; //Only 00 and 11 have c1=1. For 01 and 10, c1_=0
      c2_[i * nf_ * nf_ + f * nf_ + f] = (f == 0 ? U * (density_[0] - 0.5) : (-U * (density_[3] - 0.5)));
      //density_ are stored as density_[0],density_[1],density_[2],density_[3]
    }
  }
}

void fermionic_green_function::set_hubbard_hifreq_tails(alps::hdf5::archive &ar) {
  double U;
  ar >> alps::make_pvp("/parameters/U", U);
  set_hubbard_hifreq_tails(U);
}

void fermionic_green_function::read_density(alps::hdf5::archive &ar, bool print) {
  if (print)
    std::cout << "reading single particle density from density estimator" << std::endl;

  std::fill(density_.begin(), density_.end(), 0.);
  double sign;
  ar >> alps::make_pvp("/simulation/results/Sign/mean/value", sign);

  std::vector<double> d_buff;
  for (int i = 0; i < nf_; ++i) {
    for (int j = 0; j < nf_; ++j) {
      std::stringstream buff_name;
      buff_name
        << "/simulation/results/densities_" << i << "_" << j << "_times_sign/mean/value";
      ar >> alps::make_pvp(buff_name.str(), d_buff);
      for (int n = 0; n < ns_; ++n) {
        density_[i * nf_ + j] += d_buff[n * ns_ + n];
      }
      density_[i * nf_ + j] /= (ns_ * sign);
    }
  }
  set_hubbard_hifreq_tails(ar);
}

void fermionic_green_function::read_single_freq(alps::hdf5::archive &ar, bool print) {
  /**
   * IO function for single particle Green's function.
   * receives the CT-AUX sim hdf file and reads data from
   * <CODE>/simulation/results/G_omega_up_re"<<k<<"/mean/value</CODE> and related locations.
   * In addition reads density data out of <CODE>/simulation/results/density_up/mean/value</CODE>
   *
   */
  std::string e_or_v = to_string(e_or_v_);
  if (print)
    std::cout << "reading sign from sim.h5 file" << std::endl;
  double sign_result;
  ar >> alps::make_pvp("/simulation/results/Sign/mean/value", sign_result);
  if (print) {
    std::cout << "Using value of Sign = " << sign_result << std::endl;
    std::cout << "reading single particle GF from single-frequency estimator" << std::endl;
  }

  //In the old ALPS, single frequency GF is a signed observable, here we will need to divide the value in sim.h5 by sign

  if (conv_ != beta_conv_single_freq) throw std::logic_error("this function reads single frequency GF");

  std::vector<double> G_re_mean, G_im_mean;
  for (int K = 0; K < ns_; ++K) {
    for (int i = 0; i < nf_; ++i) {
      for (int j = 0; j < nf_; ++j) {
        std::stringstream G_re_name, G_im_name;
        G_re_name << "/simulation/results/G_omega_re" << K << "_" << i << "_" << j << "_times_sign/mean/" << e_or_v;
        G_im_name << "/simulation/results/G_omega_im" << K << "_" << i << "_" << j << "_times_sign/mean/" << e_or_v;

        ar >> alps::make_pvp(G_re_name.str(), G_re_mean);
        ar >> alps::make_pvp(G_im_name.str(), G_im_mean);

        for (int omega = 0; omega < nt_; ++omega) {
          operator()(omega, K, i, j) = std::complex<double>(G_re_mean[omega], G_im_mean[omega]) / sign_result;
        }
      }
    }
  }
  read_density(ar);
}

void fermionic_green_function::read_two_freq(alps::hdf5::archive &ar, bool print) {
  /**
   * IO function for single particle Green's function.
   * receives the CT-AUX sim hdf file and reads data from
   * <CODE>/simulation/results/G_omega_omega_up_re"<<k<<"/mean/value</CODE> and related locations.
   * This function does not read the density and it assumes that the Green's function is defined as a two-frequency GF.
   *
   */
  std::string e_or_v = to_string(e_or_v_);

  if (print)
    std::cout << "reading sign from sim.h5 file" << std::endl;
  double sign_result;
  ar >> alps::make_pvp("/simulation/results/Sign/mean/value", sign_result);
  if (print) {
    std::cout << "Using value of Sign = " << sign_result << std::endl;
    std::cout << "reading single particle GF from two-frequency estimator" << std::endl;
  }

  if (conv_ != beta_conv_two_freq) throw std::logic_error("this function reads two frequency GF");

  std::vector<double> G_re_mean, G_im_mean;
  for (int K = 0; K < ns_; ++K) {
    for (int i = 0; i < nf_; ++i) {
      for (int j = 0; j < nf_; ++j) {
        std::stringstream G_re_name, G_im_name;
        G_re_name
          << "/simulation/results/G_omega_omega_" << i << j << "_" << "re" << K
          << "_" << K << "_times_sign/mean/" << e_or_v;
        G_re_name
          << "/simulation/results/G_omega_omega_" << i << j << "_" << "im" << K
          << "_" << K << "_times_sign/mean/" << e_or_v;

        ar >> alps::make_pvp(G_re_name.str(), G_re_mean);
        ar >> alps::make_pvp(G_im_name.str(), G_im_mean);

        for (int omega = 0; omega < nt_; ++omega) {
          int index = 2 * nt_ * (omega + nt_) + omega + nt_;
          operator()(omega, K, i, j)
            = std::complex<double>(G_re_mean[index], G_im_mean[index]) / (beta_ * ns_ * sign_result);
        }
      }
    }
  }
}

} // namespace nambu
} // namespace SCinverter