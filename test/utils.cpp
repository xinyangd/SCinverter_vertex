
#include "catch.hpp"
#include "type.h"
#include "utils.h"

TEST_CASE ("Matrix dump and load", "[hdf5]") {
  using namespace SCinverter;

  int N = 3, M = 4;
  Matrix<double> x = Matrix<double>::Random(N, M);
  Matrix<std::complex<double> > z = Matrix<std::complex<double> >::Random(N, M);

  alps::hdf5::archive ar("test.h5", "w");
  ar["matrix_x"] << x;
  ar["matrix_z"] << z;
  ar.close();

  ar.open("test.h5", "r");
  Matrix<double> x1;
  Matrix<std::complex<double> > z1;

  ar["matrix_x"] >> x1;
  ar["matrix_z"] >> z1;
  ar.close();

  CHECK((x1 - x).norm() < 1e-9);
  CHECK((z1 - z).norm() < 1e-9);

  Vector<std::complex<double> > zv = Vector<std::complex<double> >::Random(N);
  ar.open("test.h5", "w");
  ar["vector_z"] << zv;
  ar.close();

  ar.open("test.h5", "r");
  Vector<std::complex<double> > zv1;

  ar["vector_z"] >> zv1;
  ar.close();

  CHECK((zv1 - zv).norm() < 1e-9);

  Tensor<std::complex<double>, 3> A(3, 4, 5);
  A.set_zero();
  Tensor<std::complex<double>, 3> B(3, 4, 5);
  B.set_zero();
  A(1).matrix().block(0, 0, 3, 4) = z;

  ar.open("test.h5", "w");
  ar["matrix_Q"] << A(1).matrix();
  ar["tensor"] << A(1);
  ar.close();

  ar.open("test.h5", "r");
  auto Bm = B(1).matrix();
  ar["matrix_Q"] >> Bm;
  auto Bt = B(2);
  ar["tensor"] >> Bt;

  CHECK((A(1).matrix() - B(1).matrix()).norm() < 1e-9);
  CHECK((A(1).matrix() - B(2).matrix()).norm() < 1e-9);
}

TEST_CASE ("Tensor View", "[view]") {
  using namespace SCinverter;

  Tensor<std::complex<double>, 3> A(3, 4, 5);
  A.set_zero();
  auto a = A(1, 2);
  Tensor_VecView(a, 5) = Vector<std::complex<double> >::Random(5);
  std::cout << Tensor_VecView(A(1, 2), 5) << std::endl;
}
