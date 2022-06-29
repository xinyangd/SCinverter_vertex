
#include <alps/params.hpp>
#include "parameters.h"
#include "fluctuation_task.h"

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  alps::mpi::communicator comm;

  if(!comm.rank()) {
    std::cout << "Running SC inverter on "<< comm.size() << " cores." << std::endl;
    auto start_time = std::chrono::system_clock::now();
    std::cout << "Start time = " << print_hr_time(start_time) << std::endl;
  }

  alps::params p(argc, argv);
  SCinverter::nambu::define_parameters(p);
  SCinverter::nambu::initialize_parameters(p);

  p.define<bool>("SINGLE_CORE", 1, "use single core to get susceptibility");
  p.define<bool>("ALT", 0, "use different SD equation");

  if(!comm.rank()){
    if(p.help_requested(std::cout)) {
      p.print_help(std::cout);
      return 0;
    }
    std::cout << p << std::endl;
  }

  fluctuation_task sigma_task(MPI_COMM_WORLD, p);
  sigma_task.work();
  sigma_task.save_results();

  if(!comm.rank()){
    auto end_time = std::chrono::system_clock::now();
    std::cout << "End time = " << print_hr_time(end_time) << std::endl;
  }

  MPI_Finalize();
}