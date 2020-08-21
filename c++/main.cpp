#include <iostream>
#include <string>
#include <memory>

#include "grid.hpp"

int main(int agrc, char *argv[]) {
    std::string initial_conditions = "REANALYSIS";
    std::string orographies = "EARTH_OROGRAPHY";
    
    std::unique_ptr<Grid> grid = std::make_unique<Grid>(initial_conditions, orographies);
    
    grid->load_initial_values();
    
    int nt = grid->nt, timesteps_between_outputs = grid->timesteps_between_outputs, i_save = 0;
    for (int n = 0; n < nt; ++n) {
        if (n % timesteps_between_outputs == 0) {
            double max_u = grid->max_u(grid->u, grid->v);
            std::cout << "Time = " << n / timesteps_between_outputs << ", max windspeed = " << max_u << "\n";
            grid->store_snapshot(i_save, n / timesteps_between_outputs);
            i_save++;
        }
        grid->compute_accleration();
        grid->lax_wendroff();
        grid->update_inner_values();
        grid->update_boudary();
    }
    
    return 0; 
}