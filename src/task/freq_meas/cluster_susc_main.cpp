
#include <mpi.h>
#include <alps/params.hpp>
#include "parameters.h"
#include "cluster_susc_task.h"

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  alps::mpi::communicator comm;

  if(!comm.rank()){
    std::cout << "Running SC inverter on "<< comm.size() << " cores." << std::endl;
  }

  alps::params p(argc, argv);
  SCinverter::nambu::define_parameters(p);
  SCinverter::nambu::initialize_parameters(p);

  if(!comm.rank()){
    if(p.help_requested(std::cout)) {
      p.print_help(std::cout);
      return 0;
    }
    std::cout << p << std::endl;
  }

  cluster_susc_task ph_task(MPI_COMM_WORLD, p, vert_channel_enum::ph);
  ph_task.get_susc();

  MPI_Finalize();
}