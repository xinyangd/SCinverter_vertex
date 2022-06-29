
#include "cluster_susc_task.h"

cluster_susc_task::cluster_susc_task(MPI_Comm comm, const alps::params &p,
                                     vert_channel_enum ch_phpp, bool compute_full_chi0) :
  task(p), comm_(comm), chi_file_path_(p["CHI_FILE_PATH"].as<std::string>()), compute_full_chi0_(compute_full_chi0),
  n_omega4_bose_(p_["NOMEGA4_BOSE"]), n_omega4_(p_["NOMEGA4"]), n_gf_(n_freq_ - n_omega4_bose_),
  channel_phpp_(ch_phpp), chi0_c_phpp_(p_, ks_, ch_phpp, p["NOMEGA4"], p["ctaux.FOURPOINT_MEAS_NUM"]),
  chi0_c_full_phpp_(p_, ks_, ch_phpp, n_gf_, p["ctaux.FOURPOINT_MEAS_NUM"]),
  chi_c_phpp_(p_, ks_, ch_phpp, p["ctaux.FOURPOINT_MEAS_NUM"]),
  sfg2_(n_freq_, n_sites_, 2, beta_, beta_conv_single_freq) {

  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);

  if (mpi_rank_ == 0) {
    sfg2_.read_single_freq_ff(p_["NAMBU_Gomega"], p_["VERBOSE"]);
  }

  MPI_Bcast(&(sfg2_.val()[0]), n_freq_ * n_sites_ * 2 * 2, MPI_DOUBLE_COMPLEX, 0, comm_);

  if (mpi_rank_ == 0) {
    sfg2_.write_single_freq(out_file_basename_ + "_sfgf_nambu", true);
    if (p_["REARRANGE"]) {
      mc_vertex g4(p_, ks_, channel_phpp_);
      temp_.open(p_["TEMP_FILE_NAME"], "w");
      g4.rearrange(temp_, ar_, p_["VERBOSE"]);
      temp_.close();
    }
  }
  MPI_Barrier(comm_);
};

void cluster_susc_task::get_susc() {

  int chi_size = chi_c_phpp_.tot_size();
  int chi0_size = chi0_c_phpp_.tot_size();
  int chi0_full_size = chi0_c_full_phpp_.tot_size();

  MPI_Status status;
  MPI_Status status1;

  int tag1 = 0;
  int tag2 = 1;

  int active_workers = mpi_size_ - 1;
  if(mpi_rank_ == 0) {
    int Q, nu;

    ar_out_.open(chi_file_path_ + out_file_basename_ + ".h5", "w");

    while(active_workers > 0) {

      MPI_Recv(&Q, 1, MPI_INT, MPI_ANY_SOURCE, tag1, comm_, &status);
      // <check if there is data. (--active_worker if received termination flag)>
      if (Q == -1) {
        active_workers --;
      }
      else {
        MPI_Recv(&nu, 1, MPI_INT, status.MPI_SOURCE, tag2, comm_, &status1);
        MPI_Recv(chi_c_phpp_.vert_tensor().data(), chi_size, MPI_DOUBLE_COMPLEX,
                 status.MPI_SOURCE, tag2, comm_, &status1);
        MPI_Recv(chi0_c_phpp_.vert_tensor().data(), chi0_size, MPI_DOUBLE_COMPLEX,
                 status.MPI_SOURCE, tag2, comm_, &status1);
        if (compute_full_chi0_)
          MPI_Recv(chi0_c_full_phpp_.vert_tensor().data(), chi0_full_size, MPI_DOUBLE_COMPLEX,
                   status.MPI_SOURCE, tag2, comm_, &status1);

        // <write received data into file>
        chi_c_phpp_.set_Qnu(Q, nu);
        chi0_c_phpp_.set_Qnu(Q, nu);
        if (compute_full_chi0_)
          chi0_c_full_phpp_.set_Qnu(Q, nu);

        chi_c_phpp_.write(ar_out_);
        chi0_c_phpp_.write(ar_out_);
        if (compute_full_chi0_)
          chi0_c_full_phpp_.write(ar_out_);
      }
    }
  }

  if (mpi_rank_ || (mpi_size_ == 1)) {
    int loop_start = mpi_size_ == 1 ? 0 : (mpi_rank_ - 1);
    int step = mpi_size_ == 1 ? 1 : (mpi_size_ - 1);

    temp_.open(p_["TEMP_FILE_NAME"], "r");
    for(int i = loop_start; i < n_sites_ * (2 * n_omega4_bose_ + 1); i += step) {
      int Q = i / (2 * n_omega4_bose_ + 1);
      int nu = i % (2 * n_omega4_bose_ + 1) - n_omega4_bose_;

      chi_c_phpp_.set_Qnu(Q, nu);
      chi0_c_phpp_.set_Qnu(Q, nu);
      if (compute_full_chi0_)
        chi0_c_full_phpp_.set_Qnu(Q, nu);

      chi_c_phpp_.compute_chi(temp_, sfg2_);
      chi0_c_phpp_.compute_chi0(sfg2_);
      if (compute_full_chi0_) {
        chi0_c_full_phpp_.compute_chi0(sfg2_);
      }

      if (mpi_rank_) {
        // send data to master
        MPI_Send(&Q, 1, MPI_INT, 0, tag1, comm_);
        MPI_Send(&nu, 1, MPI_INT, 0, tag2, comm_);
        MPI_Send(chi_c_phpp_.vert_tensor().data(), chi_size, MPI_DOUBLE_COMPLEX, 0, tag2, comm_);
        MPI_Send(chi0_c_phpp_.vert_tensor().data(), chi0_size, MPI_DOUBLE_COMPLEX, 0, tag2, comm_);
        if (compute_full_chi0_)
          MPI_Send(chi0_c_full_phpp_.vert_tensor().data(), chi0_full_size, MPI_DOUBLE_COMPLEX, 0, tag2, comm_);

      }
      else {
        chi_c_phpp_.write(ar_out_);
        chi0_c_phpp_.write(ar_out_);
        if (compute_full_chi0_)
          chi0_c_full_phpp_.write(ar_out_);
      }
    }
    if (mpi_rank_) {
      int stop = -1;
      // send termination flag to master
      MPI_Send(&stop, 1, MPI_INT, 0, tag1, comm_);
    }
    temp_.close();
  }

  if (mpi_rank_ == 0) {
    ar_out_.close();
  }

  MPI_Barrier(comm_);
}

void cluster_susc_task::get_susc_single() {

  if (mpi_rank_ == 0) {

    temp_.open(p_["TEMP_FILE_NAME"], "r");

    ar_out_.open(out_file_basename_ + ".h5", "w");
    for (int Q = 0; Q < n_sites_; ++ Q) {
      for (int nu = -n_omega4_bose_; nu <= n_omega4_bose_; ++nu) {
        chi_c_phpp_.set_Qnu(Q, nu);
        chi0_c_phpp_.set_Qnu(Q, nu);
        if (compute_full_chi0_)
          chi0_c_full_phpp_.set_Qnu(Q, nu);

        chi_c_phpp_.compute_chi(temp_, sfg2_);
        chi0_c_phpp_.compute_chi0(sfg2_);
        if (compute_full_chi0_) {
          chi0_c_full_phpp_.compute_chi0(sfg2_);
        }

        chi_c_phpp_.write(ar_out_);
        chi0_c_phpp_.write(ar_out_);
        if (compute_full_chi0_)
          chi0_c_full_phpp_.write(ar_out_);
      }
    }
    ar_out_.close();
    temp_.close();
  }
  MPI_Barrier(comm_);
}