
#include "full_vertex_task.h"

full_vertex_task::full_vertex_task(MPI_Comm comm, const alps::params &p) :
  task(p), comm_(comm), n_omega4_bose_(p_["NOMEGA4_BOSE"]), n_omega4_(p_["NOMEGA4"]),
  chi_file_path_(p["CHI_FILE_PATH"].as<std::string>()),
  F_file_path_(p["F_FILE_PATH"].as<std::string>()),
  F_phpp_(p_, ks_, vert_channel_enum::ph, p["ctaux.FOURPOINT_MEAS_NUM"]) {

  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);

  MPI_Barrier(comm_);
};

void full_vertex_task::get_F() {
  if (p_["COMPUTE_CHI"]) {
    compute_full_vertex_from_chi();

    if (mpi_rank_ == 0) {
      if (p_["REMOVE"])
        std::remove(p_["TEMP_FILE_NAME"].as<std::string>().c_str());
      if (p_["VERBOSE"]) {
        std::cout << "compute F from chi done" << std::endl;
      }
    }
  }
  else {
    compute_full_vertex();

    if (mpi_rank_ == 0 && p_["VERBOSE"]) {
      std::cout << "compute F done" << std::endl;
    }
  }

  MPI_Barrier(comm_);
}

void full_vertex_task::compute_full_vertex() {

  int F_size = F_phpp_.tot_size();

  MPI_Status status;
  MPI_Status status1;

  int tag1 = 0;
  int tag2 = 1;

  int active_workers = mpi_size_ - 1;
  if(mpi_rank_ == 0) {
    int Q, nu;

    ar_F_out_.open(F_file_path_ + out_file_basename_ + "_F.h5", "w");

    while(active_workers > 0) {

      MPI_Recv(&Q, 1, MPI_INT, MPI_ANY_SOURCE, tag1, comm_, &status);
      // <check if there is data. (--active_worker if received termination flag)>
      if (Q == -1) {
        active_workers --;
      }
      else {
        MPI_Recv(&nu, 1, MPI_INT, status.MPI_SOURCE, tag2, comm_, &status1);
        MPI_Recv(F_phpp_.vert_tensor().data(), F_size, MPI_DOUBLE_COMPLEX, status.MPI_SOURCE, tag2, comm_, &status1);

        // <write received data into file>
        F_phpp_.set_Qnu(Q, nu);
        F_phpp_.write_block(ar_F_out_, "F_cluster", F_phpp_.vert_tensor());
      }
    }
  }

  if(mpi_rank_ || (mpi_size_ == 1)) {

    using vert_ch = vert_channel_enum;

    chi0_cluster_phpp chi0_c_phpp_(p_, ks_, vert_ch::ph, p_["NOMEGA4"], F_phpp_.num_block());
    chi_cluster_phpp chi_c_phpp_(p_, ks_, vert_ch::ph, F_phpp_.num_block());

    alps::hdf5::archive ar_out_;
    ar_out_.open(chi_file_path_ + out_file_basename_ + ".h5", "r");

    int loop_start = mpi_size_ == 1 ? 0 : (mpi_rank_ - 1);
    int step = mpi_size_ == 1 ? 1 : (mpi_size_ - 1);

    for(int i = loop_start; i < n_sites_ * (2 * n_omega4_bose_ + 1); i += step) {
      int Q = i / (2 * n_omega4_bose_ + 1);
      int nu = i % (2 * n_omega4_bose_ + 1) - n_omega4_bose_;

      F_phpp_.set_Qnu(Q, nu);
      chi0_c_phpp_.set_Qnu(Q, nu);
      chi_c_phpp_.set_Qnu(Q, nu);

      F_phpp_.invert_F_phpp(ar_out_, chi_c_phpp_, chi0_c_phpp_);

      if (mpi_rank_) {
        // send data to master
        MPI_Send(&Q, 1, MPI_INT, 0, tag1, comm_);
        MPI_Send(&nu, 1, MPI_INT, 0, tag2, comm_);
        MPI_Send(F_phpp_.vert_tensor().data(), F_size, MPI_DOUBLE_COMPLEX, 0, tag2, comm_);
      }
      else {
        F_phpp_.write_block(ar_F_out_, "F_cluster", F_phpp_.vert_tensor());
      }
    }
    if (mpi_rank_) {
      int stop = -1;
      // send termination flag to master
      MPI_Send(&stop, 1, MPI_INT, 0, tag1, comm_);
    }
    ar_out_.close();
  }

  if (mpi_rank_ == 0) {
    ar_F_out_.close();
  }

  MPI_Barrier(comm_);
}

void full_vertex_task::compute_full_vertex_from_chi() {

  fermionic_green_function sfg2(n_freq_, n_sites_, 2, beta_, beta_conv_single_freq);
  alps::hdf5::archive temp;

  if (mpi_rank_ == 0) {
    sfg2.read_single_freq_ff(p_["NAMBU_Gomega"], p_["VERBOSE"]);
    sfg2.write_single_freq(out_file_basename_ + "_sfgf_nambu", true);
  }

  MPI_Bcast(&(sfg2.val()[0]), n_freq_ * n_sites_ * 2 * 2, MPI_DOUBLE_COMPLEX, 0, comm_);

  if (p_["REARRANGE"]) {
    rearrange_vertex(sfg2, temp);
  }

  // -- extract chi and F

  int F_size = F_phpp_.tot_size();

  chi0_cluster_phpp chi0_c(p_, ks_, vert_channel_enum::ph, p_["NOMEGA4"], F_phpp_.num_block());
  chi_cluster_phpp chi_c(p_, ks_, vert_channel_enum::ph, F_phpp_.num_block());

  MPI_Status status;
  MPI_Status status1;

  int tag1 = 0;
  int tag2 = 1;

  int active_workers = mpi_size_ - 1;
  if(mpi_rank_ == 0) {
    int Q, nu;

    ar_F_out_.open(F_file_path_ + out_file_basename_ + "_F.h5", "w");

    while(active_workers > 0) {

      MPI_Recv(&Q, 1, MPI_INT, MPI_ANY_SOURCE, tag1, comm_, &status);
      // <check if there is data. (--active_worker if received termination flag)>
      if (Q == -1) {
        active_workers --;
      }
      else {
        MPI_Recv(&nu, 1, MPI_INT, status.MPI_SOURCE, tag2, comm_, &status1);
        MPI_Recv(F_phpp_.vert_tensor().data(), F_size, MPI_DOUBLE_COMPLEX, status.MPI_SOURCE, tag2, comm_, &status1);

        // <write received data into file>
        F_phpp_.set_Qnu(Q, nu);
        F_phpp_.write_block(ar_F_out_, "F_cluster", F_phpp_.vert_tensor());
      }
    }
  }

  if(mpi_rank_ || (mpi_size_ == 1)) {

    int loop_start = mpi_size_ == 1 ? 0 : (mpi_rank_ - 1);
    int step = mpi_size_ == 1 ? 1 : (mpi_size_ - 1);

    temp.open(p_["TEMP_FILE_NAME"], "r");
    for(int i = loop_start; i < n_sites_ * (2 * n_omega4_bose_ + 1); i += step) {
      int Q = i / (2 * n_omega4_bose_ + 1);
      int nu = i % (2 * n_omega4_bose_ + 1) - n_omega4_bose_;

      chi0_c.set_Qnu(Q, nu);
      chi_c.set_Qnu(Q, nu);
      F_phpp_.set_Qnu(Q, nu);

      chi_c.compute_chi(temp, sfg2);
      chi0_c.compute_chi0(sfg2);

      F_phpp_.invert_F_phpp(chi_c, chi0_c);

      if (mpi_rank_) {
        // send data to master
        MPI_Send(&Q, 1, MPI_INT, 0, tag1, comm_);
        MPI_Send(&nu, 1, MPI_INT, 0, tag2, comm_);
        MPI_Send(F_phpp_.vert_tensor().data(), F_size, MPI_DOUBLE_COMPLEX, 0, tag2, comm_);
      }
      else {
        F_phpp_.write_block(ar_F_out_, "F_cluster", F_phpp_.vert_tensor());
      }
    }
    if (mpi_rank_) {
      int stop = -1;
      // send termination flag to master
      MPI_Send(&stop, 1, MPI_INT, 0, tag1, comm_);
    }

    temp.close();
  }

  if (mpi_rank_ == 0) {
    ar_F_out_.close();
  }

  MPI_Barrier(comm_);

}

void full_vertex_task::rearrange_vertex(const fermionic_green_function &sfg2, alps::hdf5::archive &temp) {

  if (mpi_rank_ == 0) {

    mc_vertex g4(p_, ks_, vert_channel_enum::ph);
    temp.open(p_["TEMP_FILE_NAME"], "w");
    g4.rearrange(temp, ar_, p_["VERBOSE"]);
    temp.close();
  }
  MPI_Barrier(comm_);
}