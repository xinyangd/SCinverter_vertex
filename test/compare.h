//
// Created by Xinyang Dong on 5/3/20.
//

#ifndef SCINV_COMPARE_H
#define SCINV_COMPARE_H

#include <alps/params.hpp>

#include "single_freq_gf.h"
#include "vertex.h"

using namespace SCinverter;

/// compute maximum absolute difference between v1 and v2. used in test
template <size_t N>
double max_diff(const nambu::vertex<N> &v1, const nambu::vertex<N> &v2) {
  if (v1.tot_size() != v2.tot_size()) {
    throw std::runtime_error("vector dimension mismatch");
  }

  auto a = Tensor_VecView(v1.vert_tensor(), v1.tot_size());
  auto b = Tensor_VecView(v2.vert_tensor(), v2.tot_size());
  double max_diff = (a - b).cwiseAbs().maxCoeff();
  return max_diff;
}

template <typename T, size_t N, typename C>
double max_diff(const TensorBase<T, N, C> &v1, const TensorBase<T, N, C> &v2) {
  if (v1.size() != v2.size()) {
    throw std::runtime_error("vector dimension mismatch");
  }

  auto a = Tensor_VecView(v1, v1.size());
  auto b = Tensor_VecView(v2, v2.size());
  double max_diff = (a - b).cwiseAbs().maxCoeff();
  return max_diff;
}

/// compute maximum difference to another GF
double max_abs_diff(const nambu::single_freq_gf<std::complex<double> > &gf1,
                    const nambu::single_freq_gf<std::complex<double> > &gf2) {
  if (gf1.val().size() != gf2.val().size()) {
    throw std::runtime_error("vector dimension mismatch");
  }

  double d = 0.;
  for (int i = 0; i < gf1.val().size(); ++i) {
    double m = std::abs(gf1.val()[i] - gf2.val()[i]);
    d = d > m ? d : m;
  }
  return d;
}

#endif //SCINV_COMPARE_H
