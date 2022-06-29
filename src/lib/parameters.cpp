//
// Created by Xinyang Dong on 4/25/20.
//

#include "parameters.h"

namespace SCinverter {

void define_paramfile_parameters(alps::params &p) {
  p.define<double>("t", "Hopping element for semi circular & square lattice density of states ");
  p.define<double>("tprime", 0., "Next-nearest Hopping element for square lattice density of states ");
  p.define<double>("MU", "Chemical Potential");
  p.define<int>("N", "Number of imaginary time discretization points");
  p.define<int>("NMATSUBARA", "Number of matsubara frequency discretization points");
  p.define<int>("NOMEGA4", "Number of two-particle matsubara frequency points for fermionic frequencies");
  p.define<int>("NOMEGA4_BOSE", "Number of two-particle matsubara frequency points for bosonic frequencies");
  p.define<int>("FLAVORS", 2, "Number of spins or (diagonal) orbitals");
  p.define<double>("BETA", "Inverse temperature");
  p.define<double>("H", "Staggered (antiferromagnetic) field");
  p.define<int>("ctaux.NTAU4", 0,
                "Number of tau points measured in chi tau");
  p.define<int>("ctaux.NOMEGA4",
                "Number of two-particle matsubara frequency points for fermionic frequencies, in later ctaux codes");
  p.define<int>("ctaux.NOMEGA4_BOSE",
                "Number of two-particle matsubara frequency points for bosonic frequencies, in later ctaux codes");
  p.define<std::string>("ctaux.FOURPOINT_MEAS_TYPE", "NONE", "type of four-point operator measurement");
  p.define<std::string>("ctaux.FOURPOINT_MEAS_CHANNEL", "NONE",
                        "type of four-point measurement channel, choices are ph and G_pp");
  p.define<int>("ctaux.FOURPOINT_MEAS_NUM", 1, "1 or 16, for one block or all blocks");
  p.define<int>("ctaux.FOURPOINT_MEAS_BLOCK", 0, "a number from 0 to 15");
  p.define<bool>("ctaux.VERT_SYMM", 1, "use cluster symmetry for four-point measurement or not");
  p.define<bool>("ctaux.SPARSE_GRID", 0, "if use sparse grid to measure G4(tau) or not");

  // defined in cluster.h
  define_dca_parameters(p);
}

void define_parameters(alps::params &p) {
  // parameter file parameters
  define_paramfile_parameters(p);

  //command line parameters
  p.define<int>("n_omega", -1, "modified number of fermionic frequencies for vertex");
  p.define<int>("n_Omega", -1, "modified number of bosonic frequencies for vertex");
  p.define<int>("NOMEGA4_BOSE_ORIG", -1, "number of bosonic frequencies in sim file");
  p.define<int>("NOMEGA4_ORIG", -1, "number of fermionic frequencies in sim file");

  p.define<std::string>("OUT_FILE_NAME", "vert", "base name of text output file");
  p.define<std::string>("SIM_FILE_NAME", "sim.h5", "hdf5 simulation file");
  p.define<std::string>("TEMP_FILE_NAME", "temp.h5", "hdf5 simulation file");
  p.define<bool>("REMOVE", true, "remove mc temp file");
  p.define<bool>("VERBOSE", false, "print information");

  p.define<bool>("REARRANGE", true, "rearrange mc result to get TEMP_FILE");
  p.define<bool>("COMPUTE_CHI", true, "compute chi from rearranged temp file");
  p.define<bool>("COMPUTE_F", true, "compute F from chi file");

  p.define<std::string>("CHI_FILE_PATH", "", "path to chi file");
  p.define<std::string>("F_FILE_PATH", "", "path to F file");
}

void initialize_parameters(alps::params &p) {
  // initialize two particle fermionic and bosonic frequencies
  if (p.exists("ctaux.NOMEGA4") && p.exists("ctaux.NOMEGA4_BOSE")) {
    int n_omega4_origin = p["ctaux.NOMEGA4"];
    int n_Omega4_origin = p["ctaux.NOMEGA4_BOSE"];
    p["NOMEGA4_ORIG"] = n_omega4_origin;
    p["NOMEGA4"] = n_omega4_origin;
    p["NOMEGA4_BOSE_ORIG"] = n_Omega4_origin;
    p["NOMEGA4_BOSE"] = n_Omega4_origin;

    // if specify number of frequencies needed
    if (p["n_omega"] != -1) {
      int n_omega = p["n_omega"];
      p["NOMEGA4"] = n_omega;
      if (n_omega4_origin < n_omega) {
        throw std::runtime_error("can't have more fermionic frequencies than measured!");
      }
    }
    if (p["n_Omega"] != -1) {
      int n_Omega = p["n_Omega"];
      p["NOMEGA4_BOSE"] = n_Omega;
      if (n_Omega4_origin < n_Omega) {
        throw std::runtime_error("can't have more bosonic frequencies than measured!");
      }
    }
  } else if (p["ctaux.NTAU4"].as<int>() > 0) {
    p["NOMEGA4_BOSE"] = p["n_Omega"];
  } else {
    throw std::runtime_error("can not find measured two particle frequencies!");
  }
}

namespace nambu {

void define_parameters(alps::params &p) {
  SCinverter::define_parameters(p);

  // text file input
  p.define<std::string>("NAMBU_SIGMA", "", "nambu self energy file");
  p.define<std::string>("NAMBU_Gomega", "", "nambu G omega file");
  p.define<std::string>("NAMBU_Gtau", "", "nambu G tau file");
}

void initialize_parameters(alps::params &p) {
  SCinverter::initialize_parameters(p);
}

} // namespace nambu

}// namespace SCinverter