
#ifndef INV_LIB_TYPE_H
#define INV_LIB_TYPE_H

#include <complex>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include <alps/numeric/tensors.hpp>

namespace SCinverter {

using complex = std::complex<double>;

// Tensor aliases
template <typename T, size_t N, typename S>
using TensorBase = alps::numerics::detail::tensor_base <T, N, S>;

template <typename T, size_t N>
using Tensor = alps::numerics::tensor<T, N>;

template <typename T, size_t N>
using TensorView = alps::numerics::tensor_view<T, N>;

template <typename T, size_t N, typename C>
using TensorBase = alps::numerics::detail::tensor_base<T, N, C>;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using Vector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;

template <typename T>
using MatrixMap = Eigen::Map<Matrix<T> >;
template <typename T>
using MatrixConstMap = Eigen::Map<const Matrix<T> >;

template <typename T>
using VectorMap = Eigen::Map<Vector<T> >;
template <typename T>
using VectorConstMap = Eigen::Map<const Vector<T> >;

template <typename T>
using ColVectorMap = Eigen::Map<ColVector<T> >;
template <typename T>
using ColVectorConstMap = Eigen::Map<const ColVector<T> >;


template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// collection of mc results type
enum eorv_enum {
  value, error
};

inline std::string to_string(eorv_enum c) {
  switch (c) {
    case value:
      return "value";
    case error:
      return "error";
    default:
      throw std::runtime_error("e or v unreachable");
  }
}

// collection of vertex type
enum class vert_channel_enum {
  ph, pp, dm, st
};

inline bool is_compatible(const vert_channel_enum channel1, const vert_channel_enum channel2) {
  switch (channel1) {
    case vert_channel_enum::ph: {
      return channel2 == vert_channel_enum::dm;
    }
    case vert_channel_enum::dm: {
      return channel2 == vert_channel_enum::ph;
    }
    default:
      return false;
  }
}

inline std::string to_string(vert_channel_enum c) {
  switch (c) {
    case vert_channel_enum::ph:
      return "ph";
    case vert_channel_enum::pp:
      return "pp";
    default:
      throw std::runtime_error("vert channel unreachable");
  }
}

// enumeration for vertex spin
enum class vert_spin_enum {
  ud, uu
};

inline std::string to_string(vert_spin_enum c) {
  switch (c) {
    case vert_spin_enum::ud:
      return "ud";
    case vert_spin_enum::uu:
      return "uu";
    default:
      throw std::runtime_error("vert spin unreachable");
  }
}

inline int vert_spin_enum2int(vert_spin_enum c) {
  switch (c) {
    case vert_spin_enum::ud:
      return 0;
    case vert_spin_enum::uu:
      return 1;
    default:
      throw std::runtime_error("vert spin unreachable");
  }
}

inline vert_spin_enum int2vert_spin_enum(int c) {
  switch (c) {
    case int(vert_spin_enum::ud):
      return vert_spin_enum::ud;
    case int(vert_spin_enum::uu):
      return vert_spin_enum::uu;
    default:
      throw std::runtime_error("vert spin unreachable");
  }
}

} // namespace SCinverter

#endif //INV_LIB_TYPE_H
