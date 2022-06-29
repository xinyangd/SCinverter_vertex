
#ifndef SCINV_K_SPACE_STRUCTURE_H
#define SCINV_K_SPACE_STRUCTURE_H

#include <set>
#include <alps/params.hpp>
#include "cluster.h"
#include "type.h"

class NormalStateClusterTransformer;

namespace SCinverter {

/// \brief A class for describing the momentum structure.
///
/// This class knows about how to add and subtract momenta. The necessary info is extracted from the ALPS DCA framework.
/// The class can also handle the symmetry classes of the vertex and green's functions – this, too is extracted from the ALPS DMFT framework.
class k_space_structure {
public:
  //types
  typedef std::set<size_t> site_class_type;
  typedef std::pair<size_t, size_t> pair_type;
  typedef std::set<pair_type> pair_class_type;
  typedef std::set<pair_class_type> pair_class_set_type;

  k_space_structure(const alps::params &p) : p_(p), n_sites_(p["dca.SITES"]),
                                             momenta_sum_(n_sites_, n_sites_), momenta_diff_(n_sites_, n_sites_) {
    find_vertex_symmetries();
  }

  /// return number of sites
  inline int n_sites() const { return n_sites_; }

  /// return number of dimension
  inline int dim() const { return dim_; }

  /// return corresponding momentum
  inline double momentum(int K, int d) const { return momenta_(K, d); }

  // return momentum map
  inline const Matrix<double> & all_momentum() const { return momenta_; }

  ///the momentum (0,0) is special. This gives the index that points to this momentum.
  inline int zero_momentum() const { return zero_; }

  inline int pipi_momentum() const { return pipi_; }

  /// return P + Q
  inline int momenta_sum(int P, int Q) const { return momenta_sum_(P, Q); }

  /// return P - Q
  inline int momenta_diff(int P, int Q) const { return momenta_diff_(P, Q); }

  /// return -K
  inline int momenta_neg(int P) const { return momenta_diff_(zero_, P); }

  /// access function for pair_class_set_k_space used for symmetrization of, e.g., Green's functions.
  const pair_class_set_type &pair_class_set_k_space() const { return pair_class_set_k_space_; }

  inline int symmetrized_K_index(int K, int Kprime, int Q) const {
    return vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q] / (n_sites_ * n_sites_);
  }

  inline int symmetrized_Kprime_index(int K, int Kprime, int Q) const {
    return (vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q] / n_sites_) % n_sites_;
  }

  inline int symmetrized_Q_index(int K, int Kprime, int Q) const {
    return vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q] % n_sites_;
  }

  inline int vertex_multiplicity(int K, int Kprime, int Q) const {
    return vertex_multiplicity_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q];
  }

  void precompute_dispersion_and_weights(std::vector<double> &weights, Matrix<double> &epsilon_kckl,
                                         Matrix<double> &epsilon_kckl_plus_pipi, Matrix<double> &symmetry_kckl) const;

  /// find the momenta patch a k vector belong to
  int find_2Dcluster_momentum(Vector<double> &k) const;

private:
  const alps::params &p_;
  const int n_sites_;
  int dim_;
  int zero_;
  int pipi_;

  Matrix<double> momenta_;
  Matrix<int> momenta_sum_;
  Matrix<int> momenta_diff_;

  // only useful if use vertex symmetry
  std::vector<int> vertex_symmetry_map_;
  std::vector<int> vertex_multiplicity_map_;

  pair_class_set_type pair_class_set_k_space_;

  ///find the internal symmetries of the vertex. also initialize the index of the zero momentum (0,0,....,0)
  void find_vertex_symmetries();

  /// find all momenta
  void find_momenta_table(const NambuClusterTransformer &clusterhandler);

  /// find momenta sums and differences (modulo 2 PI)
  void find_momenta_sum_diff();

  /// find zero and pipi
  void find_special_momentum();

  void find_vertex_symmetry_map(const NambuClusterTransformer &clusterhandler);

};


} // namespace SCinverter

#endif //SCINV_K_SPACE_STRUCTURE_H
