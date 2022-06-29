
#ifndef SCINV_FLUCTUATION_TASK_H
#define SCINV_FLUCTUATION_TASK_H

#include "type.h"
#include "task.h"
#include "single_freq_gf.h"
#include "F.h"

using namespace SCinverter;
using namespace SCinverter::nambu;

class fluctuation_task: public task {
public:
  fluctuation_task(MPI_Comm comm, const alps::params &p);
  virtual ~fluctuation_task() {};

  void work();

  void save_results() const;

private:
  MPI_Comm comm_;
  // number of CPUs
  int mpi_size_;
  // Current CPU id
  int mpi_rank_;

  std::string F_file_path_;

  const int n_omega4_;
  const int n_omega4_bose_;

  static constexpr int NumDyson = 2;
  static constexpr int NumSum = 8;
  static constexpr int NumChannel = 2;
  std::array <std::string, NumChannel> ch_list_;

  fermionic_green_function sfg2_;
  fermionic_self_energy sigma_;

  // hdf5 file to save all results
  alps::hdf5::archive ar_F_out_;
  alps::hdf5::archive temp_;

  F_phpp F_phpp_;

  // 1st dim \Sigam_00 or \Sigma_01, 2nd dim channels, 3rd dim index, 4th dim K, 5th dim omega
  Tensor<complex, 5> sigma_dyson_2chs_sep_;
  // 1st dim \Sigam_00 or \Sigma_01, 2nd dim channels, 3rd dim K, 4th dim omega
  Tensor<complex, 4> sigma_dyson_2chs_;

  // 1st dim \Sigam_00 or \Sigma_01, 2nd dim channels, 3rd dim index, 4th dim Q, 5th dim nu, 6th dim K, 7th dim omega
  Tensor<complex, 7> sigma_dyson_2chs_Qnu_sep_;
  // 1st dim \Sigam_00 or \Sigma_01, 2nd dim channels, 3rd dim Q, 4th dim nu, 5th dim K, 6th dim omega
  Tensor<complex, 6> sigma_dyson_2chs_Qnu_;

  // 1st dim \Sigam_00 or \Sigma_01, 2nd dim index, 3rd dim K, 4th dim omega
  Tensor<complex, 4> sigma_dyson_1ch_sep_;
  // 1st dim \Sigam_00 or \Sigma_01, 2nd dim K, 3rd dim omega
  Tensor<complex, 3> sigma_dyson_1ch_;

  complex Hartree_term_;

  void compute_F();
  void compute_sigma_dyson();

  void save_sigma_dyson(const std::string &filename, const TensorView<complex, 2> &sigma) const;
  void save_sigma(const std::string &filename, const fermionic_self_energy &sigma) const;

  void save_hdf5(const std::string &filename) const;
};

#endif //SCINV_FLUCTUATION_TASK_H
