
#include "catch.hpp"
#include "parameters.h"
#include "single_freq_gf.h"
#include "compare.h"

TEST_CASE("Test Momentum", "[momentum]") {
  using namespace SCinverter;

  alps::params p(std::string(TEST_DATA_DIR) + "UnitTests/sample_4site.param");
  define_paramfile_parameters(p);
  p["NOMEGA4"] = 1;
  p["NOMEGA4_BOSE"] = 1;

  double beta_ = p["BETA"];
  double nsite_ = p["dca.SITES"];
  double nfreq_ = p["NMATSUBARA"];
  double nflavor_ = p["FLAVORS"];

  SECTION("read and write") {
    using namespace nambu;

    fermionic_self_energy sigma(nfreq_, nsite_, 2, beta_);
    fermionic_self_energy sigma1(nfreq_, nsite_, 2, beta_);
    sigma.init_random();

    bool hr = 1;
    sigma.write_single_freq(std::string(TEST_DATA_DIR) + "temp_path", hr);
    sigma1.read_single_freq_ff(std::string(TEST_DATA_DIR) + "temp_path");

    CHECK(max_abs_diff(sigma, sigma1) < 1.e-12);
  }
}