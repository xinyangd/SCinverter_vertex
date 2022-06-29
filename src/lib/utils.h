
#ifndef SCINV_UTILS_H
#define SCINV_UTILS_H

#include <chrono>
#include <alps/hdf5.hpp>
#include "type.h"

namespace SCinverter {

template <typename T, size_t N, typename S>
inline MatrixConstMap<T> Tensor_MatView(const TensorBase<T, N, S> &tensor, int rows, int cols, int shift = 0) {
  return MatrixConstMap<T>(tensor.data() + shift, rows, cols);
};

template <typename T, size_t N, typename S>
inline MatrixMap<T> Tensor_MatView(TensorBase<T, N, S> &tensor, int rows, int cols, int shift = 0) {
  return MatrixMap<T>(tensor.data() + shift, rows, cols);
};

template <typename T, size_t N, typename S>
inline VectorConstMap<T> Tensor_VecView(const TensorBase<T, N, S> &tensor, int cols, int shift = 0) {
  return VectorConstMap<T>(tensor.data() + shift, cols);
};

template <typename T, size_t N, typename S>
inline VectorMap<T> Tensor_VecView(TensorBase<T, N, S> &tensor, int cols, int shift = 0) {
  return VectorMap<T>(tensor.data() + shift, cols);
};

template<typename F>
inline void do_one_fermi_loop(const int n_omega4, const int n_sites, F && f){
  for (int omega = -n_omega4; omega < n_omega4; ++omega) {
    for (int K = 0; K < n_sites; ++K) {
      f(K, omega);
    }
  }
}

template<typename F>
inline void do_two_fermi_loops(const int n_omega4, const int n_sites, F && f){
  for (int omega = -n_omega4; omega < n_omega4; ++omega) {
    for (int K = 0; K < n_sites; ++K) {
      for (int omegaprime = -n_omega4; omegaprime < n_omega4; ++omegaprime) {
        for (int Kprime = 0; Kprime < n_sites; ++Kprime) {
          f(K, Kprime, omega, omegaprime);
        }
      }
    }
  }
}

inline std::string print_hr_time(const std::chrono::system_clock::time_point &t){
  std::time_t tt = std::chrono::system_clock::to_time_t(t);
  struct tm * timeinfo;
  timeinfo= localtime(&tt);
  char buffer[80];
  strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
  return std::string(buffer);
}

} // namespace SCinverter

template <class T>
struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T> > : std::true_type {};

template <class T>
struct scalar_t { using type = T; };

template <typename T>
struct scalar_t<std::complex<T> > { using type = T; };

namespace alps {
namespace hdf5 {

template <typename T>
void save(archive &ar, std::string const &path,
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const &value,
          std::vector<std::size_t> /*size*/ = std::vector<std::size_t>(),
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims{size_t(value.rows()), size_t(value.cols() * (is_complex<T>::value ? 2 : 1))};
  const typename scalar_t<T>::type *value_ref = reinterpret_cast<const typename scalar_t<T>::type *> (value.data());
  ar.write(path, value_ref, dims);
}

template <typename T>
void load(archive &ar, std::string const &path,
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &value,
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims = ar.extent(path);
  value.resize(dims[0], dims[1] / (is_complex<T>::value ? 2 : 1));
  typename scalar_t<T>::type *value_ref = reinterpret_cast<typename scalar_t<T>::type *> (value.data());
  ar.read(path, value_ref, dims);
}

template <typename T>
void save(archive &ar, std::string const &path,
          Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> const &value,
          std::vector<std::size_t> /*size*/ = std::vector<std::size_t>(),
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims{size_t(value.size() * (is_complex<T>::value ? 2 : 1))};
  const typename scalar_t<T>::type *value_ref = reinterpret_cast<const typename scalar_t<T>::type *> (value.data());
  ar.write(path, value_ref, dims);
}

template <typename T>
void load(archive &ar, std::string const &path,
          Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> &value,
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims = ar.extent(path);
  value.resize(dims[0] / (is_complex<T>::value ? 2 : 1));
  typename scalar_t<T>::type *value_ref = reinterpret_cast<typename scalar_t<T>::type *> (value.data());
  ar.read(path, value_ref, dims);
}

template <typename T>
void save(archive &ar, std::string const &path,
          Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > const &value,
          std::vector<std::size_t> /*size*/ = std::vector<std::size_t>(),
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims{size_t(value.rows()), size_t(value.cols() * (is_complex<T>::value ? 2 : 1))};
  const typename scalar_t<T>::type *value_ref = reinterpret_cast<const typename scalar_t<T>::type *> (value.data());
  ar.write(path, value_ref, dims);
}

template <typename T>
void load(archive &ar, std::string const &path,
          Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > &value,
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims = ar.extent(path);
  value.resize(dims[0], dims[1] / (is_complex<T>::value ? 2 : 1));
  typename scalar_t<T>::type *value_ref = reinterpret_cast<typename scalar_t<T>::type *> (value.data());
  ar.read(path, value_ref, dims);
}

template <typename T>
void save(archive &ar, std::string const &path,
          Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> > const &value,
          std::vector<std::size_t> /*size*/ = std::vector<std::size_t>(),
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims{size_t(value.size() * (is_complex<T>::value ? 2 : 1))};
  const typename scalar_t<T>::type *value_ref = reinterpret_cast<const typename scalar_t<T>::type *> (value.data());
  ar.write(path, value_ref, dims);
}

template <typename T>
void load(archive &ar, std::string const &path,
          Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor> > &value,
          std::vector<std::size_t> chunk = std::vector<std::size_t>(),
          std::vector<std::size_t> /*offset*/ = std::vector<std::size_t>()) {
  std::vector<std::size_t> dims = ar.extent(path);
  value.resize(dims[0] / (is_complex<T>::value ? 2 : 1));
  typename scalar_t<T>::type *value_ref = reinterpret_cast<typename scalar_t<T>::type *> (value.data());
  ar.read(path, value_ref, dims);
}

} // namespace hdf5
} // namespace alps


#endif //SCINV_UTILS_H
