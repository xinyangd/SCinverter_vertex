//
// Created by Xinyang Dong on 4/26/20.
//

#ifndef SCINV_MC_VERTEX_H
#define SCINV_MC_VERTEX_H

#include <alps/params.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include "type.h"
#include "k_space_structure.h"

namespace SCinverter {
namespace nambu {

/// \brief A class for describing vertices read from Monte Carlo.
///
/// This class knows how to get vertex data from Monte Carlo.
/// It is mainly used by the vertex_phpp class to extract the \f$\chi\f$.
class mc_vertex {
public:
  mc_vertex(const alps::params &p, const k_space_structure &ks, vert_channel_enum channel, eorv_enum eorv = value)
              : ks_(ks), e_or_v_(eorv), channel_(channel), temp_file_(p["TEMP_FILE_NAME"].as<std::string>()),
                num_block_(p["ctaux.FOURPOINT_MEAS_NUM"]),
                n_omega4_ori_(p["NOMEGA4_ORIG"]), n_omega4_bose_ori_(p["NOMEGA4_BOSE_ORIG"]), n_omega4_(p["NOMEGA4"]),
                n_omega4_bose_(p["NOMEGA4_BOSE"]), n_sites_(p["dca.SITES"]), beta_(p["BETA"]),
                tot_freq_size_((2 * n_omega4_bose_ori_ + 1) * (2 * n_omega4_ori_) * (2 * n_omega4_ori_)),
                res_buffer_(tot_freq_size_) {

    if (channel_ != vert_channel_enum::ph) {
      throw std::runtime_error("only ph channel is implemented");
    }

    block_index_.resize(num_block_);
    switch(num_block_) {
      case 1: {
        block_index_ = {0};
        break;
      }
      case 16: {
        for (int ind = 0; ind < 16; ++ind) {
          block_index_[ind] = ind;
        }
        break;
      }
      default:
        throw std::runtime_error("mc vertex block num invalid");
    }

    vertex_.resize(tot_freq_size_);

  } // end constructor

  inline int freqindex3(int nu, int omega1, int omega2) const {
    return (nu + n_omega4_bose_ori_) * n_omega4_ori_ * n_omega4_ori_ * 4 + (omega1 + n_omega4_ori_) * n_omega4_ori_ * 2
           + (omega2 + n_omega4_ori_);
  };

  // fermionic multi-index
  inline int findex(int K, int omega) const { return K * (2 * n_omega4_) + (omega + n_omega4_); }

  inline std::tuple<int, int, int> shift_freq (int nu, int omega1, int omega2) const {
    return std::make_tuple(nu + n_omega4_bose_ori_, omega1 + n_omega4_ori_, omega2 + n_omega4_ori_);
  };

  inline eorv_enum e_or_v() const { return e_or_v_; };

  inline vert_channel_enum channel() const { return channel_; };

  inline int num_block() const { return num_block_; };

  inline const std::vector<int> & block_index() const { return block_index_; };

  inline const std::vector<std::complex<double> > &vert_vec() const { return vertex_; };

  // read mc results for one (Q, K, Kprime)
  void read(alps::hdf5::archive &ar, int Q, int K, int Kprime, int ind, int s, bool print = 0);

  void rearrange(alps::hdf5::archive &temp, alps::hdf5::archive &ar, bool print = 0);

private:
  const k_space_structure &ks_;
  // e_or_v should be "error" or "value"
  const eorv_enum e_or_v_;
  const vert_channel_enum channel_;

  const std::string temp_file_;

  const int num_block_;
  const int n_omega4_ori_;
  const int n_omega4_bose_ori_;
  const int n_omega4_;
  const int n_omega4_bose_;
  const int n_sites_;
  const double beta_;
  const int tot_freq_size_;

  std::vector<int> block_index_;

  /// result buffer for reading from sim.h5
  std::vector<double> res_buffer_;

  /// container to store vertex result
  std::vector<std::complex<double> > vertex_;
};

} // namespace nambu
} // namespace SCinverter

#endif //SCINV_MC_VERTEX_H
