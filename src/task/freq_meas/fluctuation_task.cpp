
#include <cstdio>
#include "full_vertex_task.h"
#include "fluctuation_task.h"

enum SD_channel {
  rho = 0,
  Sz = 1
};

fluctuation_task::fluctuation_task(MPI_Comm comm, const alps::params &p) :
  task(p), comm_(comm),
  F_file_path_(p["F_FILE_PATH"].as<std::string>()),
  n_omega4_(p_["NOMEGA4"]), n_omega4_bose_(p_["NOMEGA4_BOSE"]),
  sfg2_(n_freq_, n_sites_, 2, beta_, beta_conv_single_freq), sigma_(n_freq_, n_sites_, 2, beta_),
  F_phpp_(p_, ks_, vert_channel_enum::ph, p["ctaux.FOURPOINT_MEAS_NUM"]),
  sigma_dyson_2chs_sep_(NumDyson, NumChannel, NumSum, n_sites_, 2 * n_omega4_),
  sigma_dyson_2chs_(NumDyson, NumChannel, n_sites_, 2 * n_omega4_),
  sigma_dyson_2chs_Qnu_sep_(NumDyson, NumChannel, NumSum, n_sites_, 2 * n_omega4_bose_ + 1, n_sites_, 2 * n_omega4_),
  sigma_dyson_2chs_Qnu_(NumDyson, NumChannel, n_sites_, 2 * n_omega4_bose_ + 1, n_sites_, 2 * n_omega4_),
  sigma_dyson_1ch_sep_(NumDyson, NumSum, n_sites_, 2 * n_omega4_),
  sigma_dyson_1ch_(NumDyson, n_sites_, 2 * n_omega4_) {

  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);

  ch_list_ = {"rho", "Sz"};

  if (mpi_rank_ == 0) {
    sfg2_.read_single_freq_ff(p_["NAMBU_Gomega"], p_["VERBOSE"]);
    sfg2_.read_density(ar_);
    sigma_.read_single_freq_ff(p_["NAMBU_SIGMA"], p_["VERBOSE"]);
  }

  MPI_Bcast(&(sfg2_.val()[0]), n_freq_ * n_sites_ * 2 * 2, MPI_DOUBLE_COMPLEX, 0, comm_);
  MPI_Bcast(&(sigma_.val()[0]), n_freq_ * n_sites_ * 2 * 2, MPI_DOUBLE_COMPLEX, 0, comm_);
  MPI_Barrier(comm_);

  Hartree_term_ = U_ * (sfg2_.density()[3] - sfg2_.density()[0]) * 0.5;

  if (mpi_rank_ == 0)
    sigma_.write_single_freq(out_file_basename_ + "_sigma_nambu", 1);

  if (mpi_rank_ == 0 && p_["VERBOSE"]) {
    std::cout << "Hartree term" << std::endl;
    std::cout << "densities: " << sfg2_.density()[0] << " " << sfg2_.density()[3] << std::endl;
    std::cout << "Hartree term: " << Hartree_term_ << std::endl;

    std::cout << "fluctuation_task construction done." << std::endl;
  }
}

void fluctuation_task::work() {

  if (p_["COMPUTE_F"]) {
    compute_F();
    if (mpi_rank_ == 0) {
      if (p_["VERBOSE"])
        std::cout << "compute F done" << std::endl;
    }
  }

  compute_sigma_dyson();
}

void fluctuation_task::save_results() const {
  if (mpi_rank_ == 0) {
    if (p_["VERBOSE"])
      std::cout << "save results" << std::endl;

    // save sigma origin
    save_sigma("sigma_nambu", sigma_);

    // save sigma computed in 2 channels
    for (int i = 0; i < NumDyson; ++i) {
      for (int c = 0; c < NumChannel; ++c) {
        save_sigma_dyson(
          "sigma_dyson_2chs_" + ch_list_[c] + "_" + std::to_string(i / 2) + std::to_string(i % 2),
          sigma_dyson_2chs_(i, c));
        for (int j = 0; j < NumSum; ++j) {
          save_sigma_dyson(
            "sigma_dyson_2chs_" + ch_list_[c] + "_" + std::to_string(i / 2) + std::to_string(i % 2) + "_" +
            std::to_string(j), sigma_dyson_2chs_sep_(i, c, j));
        }
      }

      // save sigma computed in 1 channel
      save_sigma_dyson("sigma_dyson_1ch_" + std::to_string(i / 2) + std::to_string(i % 2), sigma_dyson_1ch_(i));
      for (int j = 0; j < NumSum; ++j) {
        save_sigma_dyson("sigma_dyson_1ch_" + std::to_string(i / 2) + std::to_string(i % 2) + "_" + std::to_string(j),
                         sigma_dyson_1ch_sep_(i, j));
      }
    }

    // save sigma_dyson_2chs_Qnu_sep_, sigma_dyson_2chs_Qnu_, sigma_dyson_2chs_sep_, sigma_dyson_2chs_ into a hdf5 file
    save_hdf5("sigma_dyson_2chs.h5");

  } // mpi end
}

void fluctuation_task::compute_F() {
  full_vertex_task ph_task(comm_, p_);
  ph_task.get_F();

  MPI_Barrier(comm_);
}

void fluctuation_task::compute_sigma_dyson() {

  double t1 = MPI_Wtime();

  sigma_dyson_2chs_.set_zero();
  sigma_dyson_2chs_sep_.set_zero();

  sigma_dyson_2chs_Qnu_.set_zero();
  sigma_dyson_2chs_Qnu_sep_.set_zero();

  sigma_dyson_1ch_.set_zero();
  sigma_dyson_1ch_sep_.set_zero();

  double pre_factor = U_ / (beta_ * beta_ * n_sites_ * n_sites_);

  const int ud = vert_spin_enum2int(vert_spin_enum::ud);
  const int uu = vert_spin_enum2int(vert_spin_enum::uu);

  const fermionic_green_function &sfg2 = sfg2_;
  const F_phpp &F_ph_ = F_phpp_;

  int Q, nu;

  auto compute_sigma_2chs_sep = [&](const int K, const int K1, const int omega, const int omega1) {
    
    int KpQ = ks_.momenta_sum(K, Q); //(K+Q)
    int K1pQ = ks_.momenta_sum(K1, Q); //(K'+Q)
    int mK1mQ = ks_.momenta_diff(ks_.zero_momentum(), K1pQ);
    int mK1 = ks_.momenta_diff(ks_.zero_momentum(), K1);
    int omega_shift = omega + n_omega4_;
    int nu_shift = nu + n_omega4_bose_;

    complex G00_kpq = sfg2(omega + nu, KpQ, 0, 0);
    complex G01_kpq = sfg2(omega + nu, KpQ, 0, 1);
    complex G11_mk1mq = sfg2(-omega1 - nu - 1, mK1mQ, 1, 1);
    complex G10_mk1mq = sfg2(-omega1 - nu - 1, mK1mQ, 1, 0);
    complex G11_mk1 = sfg2(-omega1 - 1, mK1, 1, 1);
    complex G01_mk1 = sfg2(-omega1 - 1, mK1, 0, 1);

    // sigma_00 density channel
    sigma_dyson_2chs_Qnu_sep_(0, rho, 0, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 0) + F_ph_(uu, K, K1, omega, omega1, 0, 0))
      * G00_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 1, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 1) + F_ph_(uu, K, K1, omega, omega1, 1, 1))
      * G00_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 2, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 1) - F_ph_(uu, K, K1, omega, omega1, 2, 1))
      * G01_kpq * G10_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 3, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 0) + F_ph_(uu, K, K1, omega, omega1, 2, 0))
      * G01_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 4, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 1) + F_ph_(uu, K, K1, omega, omega1, 3, 1))
      * G01_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 5, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 0) - F_ph_(uu, K, K1, omega, omega1, 1, 0))
      * G00_kpq * G11_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 6, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 1) - F_ph_(uu, K, K1, omega, omega1, 0, 1))
      * G00_kpq * G10_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, rho, 7, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 0) - F_ph_(uu, K, K1, omega, omega1, 3, 0))
      * G01_kpq * G11_mk1mq * G01_mk1;

    // sigma_00 magnetic channel
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 0, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 0) - F_ph_(uu, K, K1, omega, omega1, 0, 0))
      * G00_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 1, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 1) - F_ph_(uu, K, K1, omega, omega1, 1, 1))
      * G00_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 2, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 1) + F_ph_(uu, K, K1, omega, omega1, 2, 1))
      * G01_kpq * G10_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 3, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 0) - F_ph_(uu, K, K1, omega, omega1, 2, 0))
      * G01_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 4, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 1) - F_ph_(uu, K, K1, omega, omega1, 3, 1))
      * G01_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 5, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 0) + F_ph_(uu, K, K1, omega, omega1, 1, 0))
      * G00_kpq * G11_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 6, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 1) + F_ph_(uu, K, K1, omega, omega1, 0, 1))
      * G00_kpq * G10_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(0, Sz, 7, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 0) + F_ph_(uu, K, K1, omega, omega1, 3, 0))
      * G01_kpq * G11_mk1mq * G01_mk1;

    // sigma_01 density channel
    sigma_dyson_2chs_Qnu_sep_(1, rho, 0, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 2) - F_ph_(uu, K, K1, omega, omega1, 1, 2))
      * G00_kpq * G11_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 1, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 2) + F_ph_(uu, K, K1, omega, omega1, 2, 2))
      * G01_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 2, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 3) + F_ph_(uu, K, K1, omega, omega1, 3, 3))
      * G01_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 3, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 2) - F_ph_(uu, K, K1, omega, omega1, 3, 2))
      * G01_kpq * G11_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 4, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 3) - F_ph_(uu, K, K1, omega, omega1, 2, 3))
      * G01_kpq * G10_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 5, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 2) + F_ph_(uu, K, K1, omega, omega1, 0, 2))
      * G00_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 6, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 3) + F_ph_(uu, K, K1, omega, omega1, 1, 3))
      * G00_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, rho, 7, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 3) - F_ph_(uu, K, K1, omega, omega1, 0, 3))
      * G00_kpq * G10_mk1mq * G11_mk1;

    // sigma_01 magnetic channel
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 0, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 2) + F_ph_(uu, K, K1, omega, omega1, 1, 2))
      * G00_kpq * G11_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 1, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 2) - F_ph_(uu, K, K1, omega, omega1, 2, 2))
      * G01_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 2, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 3) - F_ph_(uu, K, K1, omega, omega1, 3, 3))
      * G01_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 3, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 2) + F_ph_(uu, K, K1, omega, omega1, 3, 2))
      * G01_kpq * G11_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 4, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 2, 3) + F_ph_(uu, K, K1, omega, omega1, 2, 3))
      * G01_kpq * G10_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 5, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 2) - F_ph_(uu, K, K1, omega, omega1, 0, 2))
      * G00_kpq * G11_mk1mq * G11_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 6, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 1, 3) - F_ph_(uu, K, K1, omega, omega1, 1, 3))
      * G00_kpq * G10_mk1mq * G01_mk1;
    sigma_dyson_2chs_Qnu_sep_(1, Sz, 7, Q, nu_shift, K, omega_shift) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 3) + F_ph_(uu, K, K1, omega, omega1, 0, 3))
      * G00_kpq * G10_mk1mq * G11_mk1;
  };

  std::function<void(const int, const int, const int, const int)> compute_2ch;
  compute_2ch = compute_sigma_2chs_sep;

  auto compute_sigma_1ch_sep = [&](const int K, const int K1, const int omega, const int omega1) {
    int KpQ = ks_.momenta_sum(K, Q); //(K+Q)
    int K1pQ = ks_.momenta_sum(K1, Q); //(K'+Q)
    int mKmQ = ks_.momenta_diff(ks_.zero_momentum(), KpQ);
    int mK1mQ = ks_.momenta_diff(ks_.zero_momentum(), K1pQ);

    complex G00_k1 = sfg2(omega1, K1, 0, 0);
    complex G01_k1 = sfg2(omega1, K1, 0, 1);
    complex G11_mk1mq = sfg2(-omega1 - nu - 1, mK1mQ, 1, 1);
    complex G10_mk1mq = sfg2(-omega1 - nu - 1, mK1mQ, 1, 0);
    complex G11_mkmq = sfg2(-omega - nu - 1, mKmQ, 1, 1);
    complex G01_mkmq = sfg2(-omega - nu - 1, mKmQ, 0, 1);

    // sigma_00 single channel
    sigma_dyson_1ch_sep_(0, 0, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 0) - F_ph_(uu, K, K1, omega, omega1, 0, 0))
      * G00_k1 * G11_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(0, 1, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 2, 1) + F_ph_(uu, K, K1, omega, omega1, 2, 1))
      * G00_k1 * G10_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(0, 2, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 1, 1) - F_ph_(uu, K, K1, omega, omega1, 1, 1))
      * G01_k1 * G10_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(0, 3, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 1, 0) + F_ph_(uu, K, K1, omega, omega1, 1, 0))
      * G01_k1 * G11_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(0, 4, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 1) - F_ph_(uu, K, K1, omega, omega1, 3, 1))
      * G01_k1 * G10_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(0, 5, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 2, 0) - F_ph_(uu, K, K1, omega, omega1, 2, 0))
      * G00_k1 * G11_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(0, 6, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 1) + F_ph_(uu, K, K1, omega, omega1, 0, 1))
      * G00_k1 * G10_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(0, 7, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 0) + F_ph_(uu, K, K1, omega, omega1, 3, 0))
      * G01_k1 * G11_mk1mq * G01_mkmq;

    // sigma_01 single channel
    sigma_dyson_1ch_sep_(1, 0, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 2, 2) - F_ph_(uu, K, K1, omega, omega1, 2, 2))
      * G00_k1 * G11_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(1, 1, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 1, 2) + F_ph_(uu, K, K1, omega, omega1, 1, 2))
      * G01_k1 * G11_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(1, 2, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 3) - F_ph_(uu, K, K1, omega, omega1, 3, 3))
      * G01_k1 * G10_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(1, 3, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 3, 2) + F_ph_(uu, K, K1, omega, omega1, 3, 2))
      * G01_k1 * G11_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(1, 4, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 1, 3) - F_ph_(uu, K, K1, omega, omega1, 1, 3))
      * G01_k1 * G10_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(1, 5, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 2) - F_ph_(uu, K, K1, omega, omega1, 0, 2))
      * G00_k1 * G11_mk1mq * G11_mkmq;
    sigma_dyson_1ch_sep_(1, 6, K, omega + n_omega4_) -=
      -(F_ph_(ud, K, K1, omega, omega1, 2, 3) + F_ph_(uu, K, K1, omega, omega1, 2, 3))
      * G00_k1 * G10_mk1mq * G01_mkmq;
    sigma_dyson_1ch_sep_(1, 7, K, omega + n_omega4_) -=
      (F_ph_(ud, K, K1, omega, omega1, 0, 3) + F_ph_(uu, K, K1, omega, omega1, 0, 3))
      * G00_k1 * G10_mk1mq * G11_mkmq;
  };

  // Q nu loop
  ar_F_out_.open(F_file_path_ + out_file_basename_ + "_F.h5", "r");

  int loop_start = mpi_rank_;
  int step = mpi_size_;

  for(int i = loop_start; i < n_sites_ * (2 * n_omega4_bose_ + 1); i += step) {
    Q = i / (2 * n_omega4_bose_ + 1);
    nu = i % (2 * n_omega4_bose_ + 1) - n_omega4_bose_;
    F_phpp_.set_Qnu(Q, nu);
    F_phpp_.read(ar_F_out_);

    do_two_fermi_loops(n_omega4_, n_sites_, compute_2ch);
    do_two_fermi_loops(n_omega4_, n_sites_, compute_sigma_1ch_sep);
  }

  ar_F_out_.close();

  if (mpi_size_ > 1) {
    Tensor<complex, 4> buffer(sigma_dyson_1ch_sep_);
    Tensor<complex, 7> buffer1(sigma_dyson_2chs_Qnu_sep_);
    MPI_Reduce(sigma_dyson_1ch_sep_.data(), buffer.data(), sigma_dyson_1ch_sep_.size(),
               MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm_);
    MPI_Reduce(sigma_dyson_2chs_Qnu_sep_.data(), buffer1.data(), sigma_dyson_2chs_Qnu_sep_.size(),
               MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm_);

    if (mpi_rank_ == 0) {
      sigma_dyson_1ch_sep_ = buffer;
      sigma_dyson_2chs_Qnu_sep_ = buffer1;
    }
  }

  if (mpi_rank_ == 0) {
    for (int j = 0; j < NumDyson; ++j) {
      for (int n = 0; n < NumSum; ++n) {
        sigma_dyson_1ch_(j).matrix() += sigma_dyson_1ch_sep_(j, n).matrix();
      }
      for (int c = 0; c < NumChannel; ++c) {
        for (int n = 0; n < NumSum; ++n) {
          for (int q = 0; q < n_sites_; ++q) {
            for (int u = 0; u < 2 * n_omega4_bose_ + 1; ++u) {
              sigma_dyson_2chs_sep_(j, c, n).matrix() += sigma_dyson_2chs_Qnu_sep_(j, c, n, q, u).matrix();
              sigma_dyson_2chs_Qnu_(j, c, q, u).matrix() += sigma_dyson_2chs_Qnu_sep_(j, c, n, q, u).matrix();
            }
          }
          sigma_dyson_2chs_(j, c).matrix() += sigma_dyson_2chs_sep_(j, c, n).matrix();
        }
      }
    }

    sigma_dyson_2chs_sep_ *= 0.5 * pre_factor;
    sigma_dyson_2chs_ *= 0.5 * pre_factor;
    sigma_dyson_2chs_Qnu_sep_ *= 0.5 * pre_factor;
    sigma_dyson_2chs_Qnu_ *= 0.5 * pre_factor;
    sigma_dyson_1ch_ *= pre_factor;
    sigma_dyson_1ch_sep_ *= pre_factor;

    sigma_dyson_1ch_(0).matrix() +=
      Hartree_term_ * Matrix<std::complex<double> >::Ones(n_sites_, 2 * n_omega4_);
  }

  double t2 = MPI_Wtime();

  if (p_["VERBOSE"]) {
    std::cout << "rank: " << mpi_rank_ << " compute time: " << t2 - t1 << std::endl;
  }

  MPI_Barrier(comm_);
}

void fluctuation_task::save_sigma_dyson(const std::string &filename, const TensorView<complex, 2> &sigma) const {
  std::ofstream fs(filename);
  fs << "#";
  for (int K = 0; K < n_sites_; ++K) {
    fs << " " << ks_.momentum(K, 0) << " " << ks_.momentum(K, 1);
  }
  fs << std::endl;
  for (int omega = -n_omega4_; omega < n_omega4_; ++omega) {
    fs << ((2 * omega + 1) * M_PI) / beta_;
    for (int K = 0; K < n_sites_; ++K) {
      fs << " " << sigma(K, omega + n_omega4_).real() << " " << sigma(K, omega + n_omega4_).imag();
    }
    fs << std::endl;
  }
  fs.close();
}

void fluctuation_task::save_sigma(const std::string &filename, const fermionic_self_energy &sigma) const {
  std::array<std::ofstream, NumDyson> fs = {std::ofstream(filename + "00"), std::ofstream(filename + "01")};
  for (int i = 0; i < fs.size(); ++i) {
    fs[i] << "#";
    for (int K = 0; K < n_sites_; ++K) {
      fs[i] << " " << ks_.momentum(K, 0) << " " << ks_.momentum(K, 1);
    }
    fs[i] << std::endl;
    for (int omega = 0; omega < n_omega4_; ++omega) {
      fs[i] << ((2 * omega + 1) * M_PI) / beta_;
      for (int K = 0; K < n_sites_; ++K) {
        fs[i] << " " << sigma(omega, K, i / 2, i % 2).real() << " " << sigma(omega, K, i / 2, i % 2).imag();
      }
      fs[i] << std::endl;
    }
    fs[i].close();
  }
}

void fluctuation_task::save_hdf5(const std::string &filename) const {
  alps::hdf5::archive outfile(filename, "w");

  outfile["n_sites"] << n_sites_;
  outfile["n_omega4"] << n_omega4_;
  outfile["n_omega4_bose"] << n_omega4_bose_;
  outfile["k_points"] << ks_.all_momentum();
  outfile["Hartree_term"] << Hartree_term_;
  outfile["sigma_dyson_2chs_sep"] << sigma_dyson_2chs_sep_;
  outfile["sigma_dyson_2chs"] << sigma_dyson_2chs_;
  outfile["sigma_dyson_2chs_Qnu_sep"] << sigma_dyson_2chs_Qnu_sep_;
  outfile["sigma_dyson_2chs_Qnu"] << sigma_dyson_2chs_Qnu_;

  outfile.close();
}