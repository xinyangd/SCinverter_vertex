//
// Created by Xinyang Dong on 8/31/21.
//

#ifndef SCINV_FULL_VERTEX_TASK_H
#define SCINV_FULL_VERTEX_TASK_H

#include "type.h"
#include "cluster_susc_task.h"
#include "F.h"

using namespace SCinverter;
using namespace SCinverter::nambu;

class full_vertex_task: public task {

public:
  full_vertex_task(MPI_Comm comm, const alps::params &p);

  ~full_vertex_task() override = default;

  void get_F();

private:

  MPI_Comm comm_;
  // number of CPUs
  int mpi_size_;
  // Current CPU id
  int mpi_rank_;

  const int n_omega4_bose_;
  const int n_omega4_;

  std::string chi_file_path_;
  std::string F_file_path_;

  alps::hdf5::archive ar_F_out_;

  F_phpp F_phpp_;

  void compute_full_vertex();
  void compute_full_vertex_from_chi();
  void rearrange_vertex(const fermionic_green_function &sfg2, alps::hdf5::archive &temp);
};

#endif //SCINV_FULL_VERTEX_TASK_H
