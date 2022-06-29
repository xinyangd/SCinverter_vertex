
#include "catch.hpp"
#include "parameters.h"
#include "k_space_structure.h"

TEST_CASE("Test Momentum", "[momentum]") {
  using namespace SCinverter;

  alps::params p(std::string(TEST_DATA_DIR) + "UnitTests/sample_4site.param");
  define_paramfile_parameters(p);
  p["VERBOSE"] = false;

  k_space_structure k_4site_(p);

  SECTION("negative") {
    for(int i = 0; i < k_4site_.n_sites(); ++i) {
      CHECK(i == k_4site_.momenta_neg(k_4site_.momenta_neg(i)));
    }
  }

  SECTION("sum diff") {
    for(int i = 0; i < k_4site_.n_sites(); ++i){
      for(int j = 0; j < k_4site_.n_sites(); ++j){
      CHECK(i == k_4site_.momenta_diff(k_4site_.momenta_sum(i,j), j));
      CHECK(j == k_4site_.momenta_diff(k_4site_.momenta_sum(i,j), i));
      }
    }
  }

SECTION("find momentum") {
    Vector<double> q(2);
    q(0) = M_PI/8; q(1) = M_PI/8;
    CHECK(k_4site_.find_2Dcluster_momentum(q) == 0);

    q(0) = M_PI/2 + 0.5; q(1) = M_PI;
    CHECK(k_4site_.find_2Dcluster_momentum(q) == 1);

    q(0) = -M_PI + 0.01; q(1) = -M_PI + 0.01;
    CHECK(k_4site_.find_2Dcluster_momentum(q) == 1);
  }
}
