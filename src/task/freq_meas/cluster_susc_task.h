
#ifndef SCINV_CLUSTER_SUSC_H
#define SCINV_CLUSTER_SUSC_H

#include <mpi.h>

#include "type.h"
#include "task.h"
#include "chi0_phpp.h"
#include "chi_phpp.h"

using namespace SCinverter;
using namespace SCinverter::nambu;

class cluster_susc_task : public task {
public:
  cluster_susc_task(MPI_Comm comm, const alps::params &p, vert_channel_enum ch_phpp, bool compute_full_chi0 = true);

  ~cluster_susc_task() override = default;

  void get_susc();

  void get_susc_single();

protected:
  MPI_Comm comm_;
  // number of CPUs
  int mpi_size_;
  // Current CPU id
  int mpi_rank_;

  std::string chi_file_path_;

  bool compute_full_chi0_;

  const int n_omega4_bose_;
  const int n_omega4_;
  const int n_gf_;
  const vert_channel_enum channel_phpp_;

  // hdf5 file to save all results
  alps::hdf5::archive ar_out_;
  alps::hdf5::archive temp_;

  chi0_cluster_phpp chi0_c_phpp_;
  chi0_cluster_phpp chi0_c_full_phpp_;
  chi_cluster_phpp chi_c_phpp_;

  fermionic_green_function sfg2_;
};


#endif //SCINV_CLUSTER_SUSC_H
