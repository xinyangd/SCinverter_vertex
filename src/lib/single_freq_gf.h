
#ifndef SCINV_SINGLE_FREQ_GF_H
#define SCINV_SINGLE_FREQ_GF_H

#include <string>
#include <complex>
#include <vector>
#include "type.h"

namespace SCinverter {
namespace nambu {

/// \brief Enum type to designate if Green's function is stored as single frequency or two frequencies.
///
/// The same single particle Green's functions can be stored in two different formats... which one is it?
/// Vertex functions should check this enum to make sure the right type of GF is being used.
enum beta_convention {
  ///Frequency convention of single frequency GF, which is 1/energy
    beta_conv_single_freq,
  ///Frequency convention of two frequency GF, which is 1/energy^2
    beta_conv_two_freq,
  ///Frequency convention not determined (yet)
    beta_conv_not_set
};

template <typename T>
class single_freq_gf {
public:
  // construction and destruction, assignment and copy constructor
  /// constructor: how many time slices, how many sites, how many flavors
  /// Now instead of an up and a down, there are 00, 01, 10 ,11, flavor in each dimension is 2
  single_freq_gf(int ntime, int nsite, int nflavor, double beta) : nt_(ntime), ns_(nsite), nf_(nflavor),
                                                                   ntns_(ntime * nsite), beta_(beta),
                                                                   val_(nt_ * ns_ * nf_ * nf_),
                                                                   density_(nf_ * nf_, 0.),
                                                                   c0_(ns_ * nf_ * nf_, 0.),
                                                                   c1_(ns_ * nf_ * nf_, 0.),
                                                                   c2_(ns_ * nf_ * nf_, 0.),
                                                                   c3_(ns_ * nf_ * nf_, 0.) { assert(nf_ == 2); };

  virtual ~single_freq_gf() {};

  void clear() { val_.fill(T{}); }

  //access of vectors and elements

  // size information
  /// how many flavors do we have? (flavors are usually spins, GF of different flavors are zero)
  inline const int &nflavor() const { return nf_; }

  /// return # of sites
  inline const int &nsite() const { return ns_; }

  /// return # of imaginary time values
  inline const int &ntime() const { return nt_; }

  /// return # of matsubara frequencies. Exactly equivalent to ntime().
  /// In case of a Matsubara GF 'ntime' sounds odd -> define 'nfreq' instead.
  inline const int &nfreq() const { return nt_; } //nfreq is an alias to ntime - more intuitive use for Matsubara GF

  ///get out the inverse temperature
  inline const double &beta() const { return beta_; }

  // const access to all values
  inline const std::vector <T> &val() const { return val_; }

  // const access to all values
  inline std::vector <T> &val() { return val_; }

  /// access element with given time, site, and flavor
  virtual inline T &operator()(int t, int site, int f1, int f2) {
    if (t < 0 || t >= nt_) throw std::logic_error("Here 3 GF access out of bounds!");
    return val_[t + nt_ * site + ntns_ * f1 + ntns_ * nf_ * f2];
  }

  /// access element with given time, site, and flavor (const reference)
  virtual inline const T operator()(int t, int site, int f1, int f2) const {
    if (t < 0 || t >= nt_) throw std::logic_error("const GF access out of bounds!");
    return val_[t + nt_ * site + ntns_ * f1 + ntns_ * nf_ * f2];
  }

  // get the highfreq coefficients
  /// high frequency coefficient of the \f$ \frac{1}{i \omega_n} \f$ term
  inline double &c1(int s1, int f1, int f2) { return c1_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// high frequency coefficient of the \f$ \frac{1}{i \omega_n} \f$ term
  inline const double &c1(int s1, int f1, int f2) const { return c1_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// high frequency coefficient of the \f$ \frac{1}{(i \omega_n)^2} \f$ term
  inline double &c2(int s1, int f1, int f2) { return c2_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// high frequency coefficient of the \f$ \frac{1}{(i \omega_n)^2} \f$ term
  inline const double &c2(int s1, int f1, int f2) const { return c2_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// high frequency coefficient of the \f$ \frac{1}{(i \omega_n)^3} \f$ term
  inline double &c3(int s1, int f1, int f2) { return c3_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// high frequency coefficient of the \f$ \frac{1}{(i \omega_n)^3} \f$ term
  inline const double &c3(int s1, int f1, int f2) const { return c3_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// constant high frequency coefficient (only needed for self energies etc)
  inline const double &c0(int s1, int f1, int f2) const { return c0_[s1 * nf_ * nf_ + nf_ * f1 + f2]; }

  /// high frequency coefficient of the \f$ \frac{1}{i \omega_n} \f$ term
  std::vector<double> &c1() { return c1_; }

  /// high frequency coefficient of the \f$ \frac{1}{(i \omega_n)^2} \f$ term
  std::vector<double> &c2() { return c2_; }

  /// high frequency coefficient of the \f$ \frac{1}{(i \omega_n)^3} \f$ term
  std::vector<double> &c3() { return c3_; }

  const std::vector<double> &c1() const { return c1_; }

  const std::vector<double> &c2() const { return c2_; }

  const std::vector<double> &c3() const { return c3_; }

protected:
  //const values
  const int ns_; /// number of sites
  const int nf_; /// number of flavors
  const int nt_; /// imag time points
  const int ntns_; /// nt*ns
  const double beta_;

  // the actual values and errors.
  std::vector <T> val_;
  std::vector<double> c0_, c1_, c2_, c3_;
  std::vector<double> density_;
};

/// \brief base class for green's functions and self-energies, combining the common concepts for the two.
///
/// This class stores a fermionic Green's function.
class fermionic_matsubara_function : public single_freq_gf<std::complex<double> > {
public:
  /// Constructor
  fermionic_matsubara_function(int ntime, int nsite, int nflavor, double beta) :
    single_freq_gf<std::complex < double> >(ntime, nsite, nflavor, beta) {};

  virtual ~fermionic_matsubara_function() {};

  /// element access function taking care of antisymmetry for negative elements:
  /// This function is fermionic, the imag part flips under negation of the argument,
  /// the real part stays the same: G(-Q)=(Re G(Q), -Im G(Q))
  inline const std::complex<double> operator()(int t, int site, int f1, int f2) const {
    if (t < -nt_ || t >= nt_) throw std::logic_error("Here 2 matsubara const GF access out of bounds!");
    if (t < 0) {
      return std::complex<double>(val_[-t - 1 + nt_ * site + ntns_ * f1 + ntns_ * nf_ * f2].real(),
                                  -val_[-t - 1 + nt_ * site + ntns_ * f1 + ntns_ * nf_ * f2].imag());
    } else {
      return val_[t + nt_ * site + ntns_ * f1 + ntns_ * nf_ * f2];
    }
  }

  inline std::complex<double> &operator()(int t, int site, int f1, int f2) {
    return single_freq_gf<std::complex<double> >::operator()(t, site, f1, f2);
  }

  /// write single frequency green's function or self energy to file.
  void write_single_freq(const std::string &out_file_name_gf, bool hr = 0) const;

  /// read single freq ferminonic matsubara function from file
  void read_single_freq_ff(const std::string &in_file_name_gf, bool print = 0);

  /// evaluate the high frequency behavior
  std::complex<double> hifreq(int t, int site, int f1, int f2) const;

  /// initialize to random values for testing/debugging purposes
  void init_random() { for (int i = 0; i < val_.size(); ++i) val_[i] = std::complex<double>(drand48(), drand48()); }
};

/// \brief fermionic Green's function, as a function of a fermionic frequency.
///
/// This class stores a fermionic Green's function.
class fermionic_green_function : public fermionic_matsubara_function {
public:
  // construction and destruction, assignement and copy constructor
  /// constructor: how many time slices, how many sites, how many flavors
  fermionic_green_function(int ntime, int nsite, int nflavor, double beta,
                           beta_convention conv, eorv_enum e_or_v = eorv_enum::value) :
    fermionic_matsubara_function(ntime, nsite, nflavor, beta), conv_(conv), e_or_v_(e_or_v) {}

  /// set Hubbard high frequency tails from known U
  void set_hubbard_hifreq_tails(double U);

  /// set Hubbard high frequency tails
  void set_hubbard_hifreq_tails(alps::hdf5::archive &ar);

  /// read the density from the ALPS density estimator for the total density
  void read_density(alps::hdf5::archive &ar, bool print = 0);

  /// file input function
  void read_single_freq(alps::hdf5::archive &ar, bool print = 0);

  /// file input function
  void read_two_freq(alps::hdf5::archive &ar, bool print = 0);

  /// const access to density
  const std::vector<double> &density() const { return density_; }

  /// is this a single or a double frequency Green's function, i.e. dimension 1/Energy or dimension 1/Energy**2?
  beta_convention conv() const { return conv_; }

  /// is this a value or error function
  eorv_enum e_or_v() const { return e_or_v_; };
  /*
  ///this is a test/debug function to set the GF to a local (i.e. k-independent) Bethe lattice gf.
  void initialize_to_local_bethe(double mu);
  */
private:
  const beta_convention conv_;
  const eorv_enum e_or_v_;
};

class fermionic_self_energy : public fermionic_matsubara_function {
public:
  fermionic_self_energy(int ntime, int nsite,
                        int nflavor, double beta) : fermionic_matsubara_function(ntime, nsite, nflavor, beta) {}
};

} // namespace nambu
} // namespace SCinverter

#endif //SCINV_SINGLE_FREQ_GF_H
