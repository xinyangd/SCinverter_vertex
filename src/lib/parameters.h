//
// Created by Xinyang Dong on 4/25/20.
//

#ifndef SCINV_PARAMETERS_H
#define SCINV_PARAMETERS_H

#include "cluster.h"
#include <alps/params.hpp>

namespace SCinverter {

void define_paramfile_parameters(alps::params &p);

void define_parameters(alps::params &p);

void initialize_parameters(alps::params &p);

namespace nambu {

void define_parameters(alps::params &p);

void initialize_parameters(alps::params &p);

} // namespace nambu

}// namespace SCinverter

#endif //SCINV_PARAMETERS_H
