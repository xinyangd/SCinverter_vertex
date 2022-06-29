
#include "k_space_structure.h"

namespace SCinverter {

void k_space_structure::find_vertex_symmetries() {

  NambuClusterTransformer clusterhandler(p_);

  ///creating momenta table
  dim_ = clusterhandler.dimension();

  if (dim_ > 2)
    throw std::logic_error("check that all of this indeed works for 3d.");
  //std::cout<<"dimension is: "<<dim_<<std::endl;
  find_momenta_table(clusterhandler);
  //std::cout<<"finding momenta sum diff."<<std::endl;
  find_momenta_sum_diff();
  //std::cout<<"done finding momenta sum diff."<<std::endl;
  find_special_momentum();
  //std::cout<<"finding equivalent k-vectors"<<std::endl;
  find_vertex_symmetry_map(clusterhandler);
  //std::cout<<"done finding equivalent k-vectors"<<std::endl;
}

void k_space_structure::find_momenta_table(const NambuClusterTransformer &clusterhandler) {

  momenta_.resize(n_sites_, dim_);

  for (int i = 0; i < n_sites_; ++i) {
    for (int j = 0; j < dim_; ++j) {
      momenta_(i, j) = clusterhandler.cluster_momentum(i, j);
    }
  }
}

void k_space_structure::find_momenta_sum_diff() {
  momenta_diff_.fill(-1);
  momenta_sum_.fill(-1);

  Vector<double> sum_val(dim_);
  Vector<double> diff_val(dim_);

  double tol = 1.e-14;
  for (int K = 0; K < n_sites_; ++K) {
    for (int Kprime = 0; Kprime < n_sites_; ++Kprime) {
      sum_val = momenta_.row(K) + momenta_.row(Kprime);
      diff_val = momenta_.row(K) - momenta_.row(Kprime);
      for (int i = 0; i < dim_; ++i) {
        if (sum_val(i) > M_PI + tol) sum_val(i) -= 2 * M_PI;
        if (sum_val(i) < -M_PI + tol) sum_val(i) += 2 * M_PI;
        if (diff_val(i) > M_PI + tol) diff_val(i) -= 2 * M_PI;
        if (diff_val(i) < -M_PI + tol) diff_val(i) += 2 * M_PI;
      }

      // find corresponding index val
      for (int Q = 0; Q < n_sites_; ++Q) {
        if ((sum_val - momenta_.row(Q)).norm() < tol)
          momenta_sum_(K, Kprime) = Q;
        if ((diff_val - momenta_.row(Q)).norm() < tol)
          momenta_diff_(K, Kprime) = Q;
      }

      if (momenta_sum_(K, Kprime) == -1) {
        std::cout << "K: " << momenta_.row(K) << " " << " kprime: " << momenta_.row(Kprime)
                  << " sum: " << sum_val << std::endl;
        throw std::logic_error("problem with periodicity of cluster!");
      }
      if (momenta_diff_(K, Kprime) == -1) {
        std::cout << "K: " << momenta_.row(K) << " " << " kprime: " << momenta_.row(Kprime)
                  << " diff: " << diff_val << std::endl;
        throw std::logic_error("problem with periodicity of cluster!");
      }
    }
  }
}

void k_space_structure::find_special_momentum() {
  //std::cout<<"done finding momenta sum diff."<<std::endl;
  double tol = 1.e-14;
  zero_ = -1;
  pipi_ = -1;
  for (int i = 0; i < n_sites_; ++i) {
    if (momenta_.row(i).norm() < tol) {
      zero_ = i;
    } else if ((momenta_.row(i) - M_PI * Vector<double>::Ones(dim_)).norm() < tol) {
      pipi_ = i;
    }
  }

  if (zero_ == -1)
    throw std::logic_error("could not find momentum zero.");

  if (n_sites_ != 1 && pipi_ == -1)//For paramagnetic calculation, 1 site does not have pipi vector
    throw std::logic_error("could not find AFM (pi,pi,...) vector.");
}

void k_space_structure::find_vertex_symmetry_map(const NambuClusterTransformer &clusterhandler) {
  //get the symmetries for pairs
  NambuClusterTransformer::pair_class_set_type pcst = clusterhandler.pair_class_set_k_space();
  for (auto it = pcst.begin(); it != pcst.end(); ++it) {
    k_space_structure::pair_class_type pct;
    for (auto it2 = (*it).begin(); it2 != (*it).end(); ++it2) {
      pct.insert(*it2);
    }
    pair_class_set_k_space_.insert(pct);
  }

  vertex_symmetry_map_.resize(n_sites_ * n_sites_ * n_sites_, n_sites_ * n_sites_ * n_sites_);
  vertex_multiplicity_map_.resize(n_sites_ * n_sites_ * n_sites_, 0);

  int symmetry_counter = 0;
  for (int K = 0; K < n_sites_; ++K) {
    for (int Kprime = 0; Kprime < n_sites_; ++Kprime) {
      for (int Q = 0; Q < n_sites_; ++Q) {

        if (p_["ctaux.VERT_SYMM"]) {
          // divide 2 means we only use the first four symmetries in the symmetry table for broken d_{x^2-y^2}
          // the order of symmetry in cluster helper is identity, \sigma_v, \sigma_h, C2, \sigma_d, C4, C4, \sigma_d
          for (std::size_t s = 0; s < int(clusterhandler.symmetry_table_k_space().size1() / 2); ++s) {
            vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q] =
              std::min(vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q],
                       clusterhandler.symmetry_table_k_space()(s, K) * n_sites_ * n_sites_
                       + clusterhandler.symmetry_table_k_space()(s, Kprime) * n_sites_
                       + clusterhandler.symmetry_table_k_space()(s, Q));
          }
        } else {
          vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q]
            = K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q;
        }

        vertex_multiplicity_map_[vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q]]++;
        if (vertex_symmetry_map_[K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q]
            == K * n_sites_ * n_sites_ + Kprime * n_sites_ + Q)
          symmetry_counter++;
      }
    }
  }
  if (p_["VERBOSE"])
    std::cout << "number of total K Kprime Q points: " << symmetry_counter << std::endl;

}

void k_space_structure::precompute_dispersion_and_weights(std::vector<double> &weight_kl,
                                                          Matrix<double> &epsilon_kckl,
                                                          Matrix<double> &epsilon_kckl_plus_pipi,
                                                          Matrix<double> &symmetry_kckl) const {
  NormalStateClusterTransformer ct(p_);
  epsilon_kckl.resize(n_sites_, ct.n_lattice_momenta());
  epsilon_kckl_plus_pipi.resize(n_sites_, ct.n_lattice_momenta());
  symmetry_kckl.resize(n_sites_, ct.n_lattice_momenta());
  weight_kl.resize(ct.n_lattice_momenta());

  for (size_t kc = 0; kc < n_sites_; ++kc) {
    for (size_t kl = 0; kl < ct.n_lattice_momenta(); ++kl) {
      std::vector<double> k(ct.dimension());
      std::vector<double> k_plus_pipi(ct.dimension());
      for (size_t d = 0; d < ct.dimension(); ++d) {
        k[d] = ct.lattice_momentum(kl, d) + ct.cluster_momentum(kc, d);
        k_plus_pipi[d] = ct.lattice_momentum(kl, d) + ct.cluster_momentum(kc, d) + M_PI;
      }
      epsilon_kckl(kc, kl) = ct.epsilon(k);
      epsilon_kckl_plus_pipi(kc, kl) = ct.epsilon(k_plus_pipi);
      symmetry_kckl(kc, kl) = cos(k[0]) - cos(k[1]); // cos(kx) - cos(ky)
    }
  }

  for (size_t kl = 0; kl < ct.n_lattice_momenta(); ++kl) {
    weight_kl[kl] = ct.lattice_momentum(kl, ct.dimension());
    //this is dangerous and may fail if we don't have 'L' explicitly specified in the parameter file.
    double L = p_["dca.L"];
    weight_kl[kl] /= L * L; //normalization to one.
  }
}

int k_space_structure::find_2Dcluster_momentum(Vector<double> &k) const {

  assert(dim_ == 2);
  assert(k.size() == dim_);

  double tol = 1.e-15;
  double distance = 1000, tmp_distance = 1000;

  int the_K = 0;
  for (int K = 0; K < n_sites_; ++K) {
    tmp_distance = (k - momenta_.row(K)).norm() * (k - momenta_.row(K)).norm();

    if (tmp_distance < distance) {
      the_K = K;
      distance = tmp_distance;
    }

    // (pi, x) also means (-pi, x)
    if (std::abs(momenta_(K, 0) - M_PI) < tol) {
      tmp_distance = (k(0) + M_PI) * (k(0) + M_PI) + (k(1) - momenta_(K, 1)) * (k(1) - momenta_(K, 1));
      if (tmp_distance < distance) {
        the_K = K;
        distance = tmp_distance;
      }
    }
    // (x, pi) also means (-x, pi)
    if (std::abs(momenta_(K, 1) - M_PI) < tol) {
      tmp_distance = (k(0) - momenta_(K, 0)) * (k(0) - momenta_(K, 0)) + (k(1) + M_PI) * (k(1) + M_PI);
      if (tmp_distance < distance) {
        the_K = K;
        distance = tmp_distance;
      }
    }
    // (pi, pi) also means (-pi, -pi)
    if (std::abs(momenta_(K, 0) - M_PI) < tol && std::abs(momenta_(K, 1) - M_PI) < tol) {
      tmp_distance = (k(0) + M_PI) * (k(0) + M_PI) + (k(1) + M_PI) * (k(1) + M_PI);
      if (tmp_distance < distance) {
        the_K = K;
        distance = tmp_distance;
      }
    }
  }
  return the_K;
}

} // namespace SC inverter