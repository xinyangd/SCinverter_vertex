
#include "catch.hpp"
#include "parameters.h"
#include "F.h"
#include "compare.h"

TEST_CASE("Test Full Vertex", "[fullvert]") {

  using namespace SCinverter;

  alps::params p(std::string(TEST_DATA_DIR) + "UnitTests/sample_4site.param");
  define_parameters(p);

  int n_omega4_ = 2;
  int n_omega4_bose_ = 2;
  p["NOMEGA4"] = n_omega4_;
  p["NOMEGA4_BOSE"] = n_omega4_bose_;
  p["ctaux.NOMEGA4"] = n_omega4_;
  p["ctaux.NOMEGA4_BOSE"] = n_omega4_bose_;
  p["ctaux.FOURPOINT_MEAS_NUM"] = 16;

  initialize_parameters(p);

  double nsites_ = p["dca.SITES"];

  k_space_structure k_4site_(p);

  vert_channel_enum channel_ = vert_channel_enum::ph;

  SECTION("ChiToFvsFtoChi") {
    using namespace nambu;

    std::string file_name_ = "vert";

    chi_cluster_phpp chi_(p, k_4site_, channel_);
    chi_cluster_phpp chi1_(p, k_4site_, channel_);
    chi0_cluster_phpp chi0_(p, k_4site_, channel_, p["NOMEGA4"]);
    chi0_cluster_phpp chi01_(p, k_4site_, channel_, p["NOMEGA4"]);
    F_phpp F_(p, k_4site_, channel_);
    F_phpp F1_(p, k_4site_, channel_);

    chi_cluster_phpp chiF_(p, k_4site_, channel_);

    alps::hdf5::archive ar_(file_name_ + ".h5", "w");
    ar_.close();

    for (int Q = 0; Q < nsites_; ++Q) {
      for (int nu = -int(p["NOMEGA4_BOSE"]); nu <= int(p["NOMEGA4_BOSE"]); ++nu) {
        chi_.init_random();
        chi_.set_Qnu(Q, nu);
        chi1_.set_Qnu(Q, nu);
        chi1_.vert_tensor() = chi_.vert_tensor();

        chi0_.init_random();
        chi0_.set_Qnu(Q, nu);
        chi01_.set_Qnu(Q, nu);
        F_.set_Qnu(Q, nu);
        F1_.set_Qnu(Q, nu);
        chiF_.set_Qnu(Q, nu);

        ar_.open(file_name_ + ".h5", "w");
        chi_.write(ar_);
        chi0_.write(ar_);
        ar_.close();

        ar_.open(file_name_ + ".h5", "r");
        F_.invert_F_phpp(ar_, chi1_, chi0_);
        ar_.close();

        ar_.open(file_name_ + ".h5", "w");
        F_.write_block(ar_, "F_cluster", F_.vert_tensor());
        ar_.close();

        ar_.open(file_name_ + ".h5", "r");
        F1_.compute_chi_from_F(ar_, chiF_, chi01_);
        ar_.close();

        ar_.open(file_name_ + ".h5", "w");
        F_.write_block(ar_, "chi_cluster", chiF_.vert_tensor());
        ar_.close();

        ar_.open(file_name_ + ".h5", "r");
        chiF_.read(ar_);
        ar_.close();

        REQUIRE(max_diff(F_, F1_) < 1e-8); // check read and write block
        REQUIRE(max_diff(chi0_, chi01_) < 1e-8);
        REQUIRE(max_diff(chi_, chiF_) < 1e-8);
      }
    }
  } // section end

  SECTION("ChiToFvsFtoChiNormal") {
    using namespace nambu;

    std::string file_name_ = "vert_Normal";
    const int BNum = 1;

    chi_cluster_phpp chi_(p, k_4site_, channel_, BNum);
    chi_cluster_phpp chi1_(p, k_4site_, channel_, BNum);
    chi0_cluster_phpp chi0_(p, k_4site_, channel_, p["NOMEGA4"], BNum);
    chi0_cluster_phpp chi01_(p, k_4site_, channel_, p["NOMEGA4"], BNum);
    F_phpp F_(p, k_4site_, channel_, BNum);
    F_phpp F1_(p, k_4site_, channel_, BNum);

    chi_cluster_phpp chiF_(p, k_4site_, channel_, BNum);

    alps::hdf5::archive ar_(file_name_ + ".h5", "w");
    ar_.close();

    for (int Q = 0; Q < nsites_; ++Q) {
      for (int nu = -int(p["NOMEGA4_BOSE"]); nu <= int(p["NOMEGA4_BOSE"]); ++nu) {
        chi_.init_random();
        chi_.set_Qnu(Q, nu);
        chi1_.set_Qnu(Q, nu);
        chi1_.vert_tensor() = chi_.vert_tensor();

        chi0_.init_random();
        chi0_.set_Qnu(Q, nu);
        chi01_.set_Qnu(Q, nu);
        F_.set_Qnu(Q, nu);
        F1_.set_Qnu(Q, nu);
        chiF_.set_Qnu(Q, nu);

        ar_.open(file_name_ + ".h5", "w");
        chi_.write(ar_);
        chi0_.write(ar_);
        ar_.close();

        ar_.open(file_name_ + ".h5", "r");
        F_.invert_F_phpp(ar_, chi1_, chi0_);
        ar_.close();

        ar_.open(file_name_ + ".h5", "w");
        F_.write_block(ar_, "F_cluster", F_.vert_tensor());
        ar_.close();

        ar_.open(file_name_ + ".h5", "r");
        F1_.compute_chi_from_F(ar_, chiF_, chi01_);
        ar_.close();

        ar_.open(file_name_ + ".h5", "w");
        F_.write_block(ar_, "chi_cluster", chiF_.vert_tensor());
        ar_.close();

        ar_.open(file_name_ + ".h5", "r");
        chiF_.read(ar_);
        ar_.close();

        REQUIRE(max_diff(F_, F1_) < 1e-8); // check read and write block
        REQUIRE(max_diff(chi0_, chi01_) < 1e-8);
        const int uu = vert_spin_enum2int(vert_spin_enum::uu);
        const int ud = vert_spin_enum2int(vert_spin_enum::ud);
        REQUIRE(max_diff(chi_.vert_tensor()(uu), chiF_.vert_tensor()(uu)) < 1e-8);
        REQUIRE(max_diff(chi_.vert_tensor()(ud), chiF_.vert_tensor()(ud)) < 1e-8);
      }
    }
  } // section end
}