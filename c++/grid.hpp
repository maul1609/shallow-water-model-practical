#include <iostream>
#include <cmath>
#include <fstream>

struct Grid {
    std::string initial_conditions;
    std::string orographies;

    // SECTION 1: Configuration
    double f = 1e-4; // Coriolis parameter (s-1)
    double g = 9.81; // Acceleration due to gravity (m/s2)
    double beta = 1.6e-11; // Meridional gradient of f (s-1m-1)

    double dt_mins = 1.0;  // Timestep (minutes)
    double output_interval_mins = 60.0; // Time between outputs (minutes)
    double forecast_length_days = 4.0; // Total simulation length (days)
    
    bool initially_geostrophic = false; // Can be "True" or "False"
    bool add_random_height_noise = false; // Can be "True" or "False"

    /* If you change the number of gridpoints then orography=EARTH_OROGRAPHY
     or initial_conditions=REANALYSIS won't work */
    int nx = 254; // Number of zonal gridpoints
    int ny = 50; // Number of meridional gridpoints

    double dx=100.0e3; // Zonal grid spacing (m)
    double dy=dx;      // Meridional grid spacing

    // SECTION 2: Act on the configuration information
    double dt = dt_mins*60.0; // Timestep (s)
    double output_interval = output_interval_mins*60.0; // Time between outputs (s)
    double forecast_length = forecast_length_days*24.0*3600.0; // Forecast length (s)
    int nt = forecast_length/dt+1; // Number of timesteps
    int timesteps_between_outputs = output_interval/dt;
    // TODO
    int noutput = nt/timesteps_between_outputs + 1; // Number of output frames
    

    
    double *X; // matrices of the coordinate variables
    double *Y; // matrices of the coordinate variables
    double *H; // the orography field "H"
    double *height; // the height of the upper surface)

    double *F; // Coriolis parameter
    double *u, *v; // wind speed matrices in x and y direction
    double *h; // Define h as the depth of the fluid
    
    double *u_save, *v_save, *h_save, *t_save;
    double *u_accel, *v_accel;

    double *uh, *vh, *Ux, *Uy, *Vx, *Vy;
    double *h_mid_xt, *h_mid_yt, *uh_mid_xt, *uh_mid_yt, *vh_mid_xt, *vh_mid_yt;
    double *Ux_mid_xt, *Uy_mid_yt, *Vx_mid_xt, *Vy_mid_yt;
    double *h_new, *uh_new, *vh_new, *u_new, *v_new;

    int i_save = 0; // Index to stored data

    Grid(std::string initial_conditions, std::string orographies)
        : initial_conditions {initial_conditions}
        , orographies {orographies} {
        
        // allocate memory
        H = new double[nx * ny];
        height = new double[nx * ny];

        X = new double[nx * ny];
        Y = new double[nx * ny];
        F = new double[nx * ny];
        u = new double[nx * ny];
        v = new double[nx * ny];
        h = new double[nx * ny];
        u_accel = new double[nx * ny];
        v_accel = new double[nx * ny];
        uh = new double[nx * ny];
        vh = new double[nx * ny];
        Ux = new double[nx * ny];
        Uy = new double[nx * ny];
        Vx = new double[nx * ny];
        Vy = new double[nx * ny];
        h_mid_xt = new double[nx * ny];
        h_mid_yt = new double[nx * ny];
        uh_mid_xt = new double[nx * ny];
        uh_mid_yt = new double[nx * ny];
        vh_mid_xt = new double[nx * ny];
        vh_mid_yt = new double[nx * ny];
        
        Ux_mid_xt = new double[nx * ny];
        Uy_mid_yt = new double[nx * ny];
        Vx_mid_xt = new double[nx * ny];
        Vy_mid_yt = new double[nx * ny];

        h_new = new double[nx * ny];
        uh_new = new double[nx * ny];
        vh_new = new double[nx * ny];
        u_new = new double[nx * ny];
        v_new = new double[nx * ny];

        u_save = new double[nx * ny * noutput];
        v_save = new double[nx * ny * noutput];
        h_save = new double[nx * ny * noutput];
        t_save = new double[noutput];

        for (int i = 0; i < nx * ny * noutput; ++i) {
            u_save[i] = 0.0;
            v_save[i] = 0.0;
            h_save[i] = 0.0;
        }
    }


    
    void load_initial_values() {
        std::cout << "initial conditions: " << initial_conditions << " orography:" << orographies << "\n";
        if (initial_conditions.compare("UNIFORM_WESTERLY") == 0 && orographies.compare("FLAT") == 0) {
            // initial_conditions == UNIFORM_WESTERLY
            double mean_wind_speed = 20.0;
            double meanY = (ny - 1) * dy / 2.0;

            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    // Create X, Y coordinate matrices
                    X[i * ny + j] = i * dx;
                    Y[i * ny + j] = j * dy;
                    // orography == FLAT
                    H[i * ny + j] = 0.0;
                    // initial_conditions == UNIFORM_WESTERLY
                    height[i * ny + j] = 10000.0 - (mean_wind_speed * f / g) * (Y[i * ny + j] - meanY);
                    // Coriolis parameter as a matrix of values varying in y only
                    F[i * ny + j] = f + beta * (Y[i * ny + j] - meanY);
                    // Initialize the wind to rest
                    u[i * ny + j] = 0.0;
                    v[i * ny + j] = 0.0;
                    // Define h as the depth of the fluid (whereas "height" is the height of the upper surface)
                    h[i * ny + j] = height[i * ny + j] - H[i * ny + j];
                }
            }
        } else if (initial_conditions.compare("REANALYSIS") == 0 && orographies.compare("EARTH_OROGRAPHY") == 0) {
            // read H and height from files
            std::ifstream H_file, height_file;
            H_file.open("H.txt");
            height_file.open("height.txt");
            if (!H_file || !height_file) {
                std::cout << "unable to open files\n";
                std::exit(1);
            }

            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    H_file >> H[i * ny + j];
                    height_file >> height[i * ny + j];
                }
            }
            
            double meanY = (ny - 1) * dy / 2.0;
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    // Create X, Y coordinate matrices
                    X[i * ny + j] = i * dx;
                    Y[i * ny + j] = j * dy;
                    
                    // Coriolis parameter as a matrix of values varying in y only
                    F[i * ny + j] = f + beta * (Y[i * ny + j] - meanY);
                    // Initialize the wind to rest
                    u[i * ny + j] = 0.0;
                    v[i * ny + j] = 0.0;
                    // Define h as the depth of the fluid (whereas "height" is the height of the upper surface)
                    h[i * ny + j] = height[i * ny + j] - H[i * ny + j];
                }
            }
        } else {
            std::cout << "unknown initial conditions and orography\n";
            std::abort();
        }
        

    }
    

    void compute_accleration() {
        for (int i = 1; i < nx -1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                u_accel[i * ny + j] = F[i * ny + j] * v[i * ny + j]
                                    - (g / (2.0 * dx)) * (H[(i + 1) * ny + j] - H[(i - 1) * ny + j]);
                v_accel[i * ny + j] = -F[i * ny + j] * u[i * ny + j]
                                    - (g / (2.0 * dy)) * (H[i * ny + j + 1] - H[i * ny + j - 1]);
            }
        }
    }

    // This function performs one timestep of the Lax-Wendroff scheme
    // applied to the shallow water equations
    void lax_wendroff() {
        //dx, dy, dt, g, u, v, h, u_accel, v_accel
        // First work out mid-point values in time and space
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                uh[i * ny + j] = u[i * ny + j] * h[i * ny + j];
                vh[i * ny + j] = v[i * ny + j] * h[i * ny + j];
                Ux[i * ny + j] = uh[i * ny + j] * u[i * ny + j]
                               + 0.5 * g * h[i * ny + j] * h[i * ny + j];
                Uy[i * ny + j] = uh[i * ny + j] * v[i * ny + j];
                Vx[i * ny + j] = Uy[i * ny + j];
                Vy[i * ny + j] = vh[i * ny + j] * v[i * ny + j]
                               + 0.5 * g * h[i * ny + j] * h[i * ny + j];
            }
        }
        
        for (int i = 1; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                h_mid_xt[i * ny + j] = 0.5 * (h[i * ny + j] + h[(i - 1) * ny + j]) \
                                     - 0.5 * dt / dx *  (uh[i * ny + j] - uh[(i - 1) * ny + j]);
                uh_mid_xt[i * ny + j] = 0.5 * (uh[i * ny + j] + uh[(i - 1) * ny + j])
                                     - 0.5 * dt / dx *  (Ux[i * ny + j] - Ux[(i - 1) * ny + j]);
                vh_mid_xt[i * ny + j] = 0.5 * (vh[i * ny + j] + vh[(i - 1) * ny + j])
                                     - 0.5 * dt / dx *  (Vx[i * ny + j] - Vx[(i - 1) * ny + j]);
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                h_mid_yt[i * ny + j] = 0.5 * (h[i * ny + j] + h[i * ny + j - 1])
                                     - 0.5 * dt / dy *  (vh[i * ny + j] - vh[i * ny + j - 1]);
                uh_mid_yt[i * ny + j] = 0.5 * (uh[i * ny + j] + uh[i * ny + j - 1])
                                     - 0.5 * dt / dy *  (Uy[i * ny + j] - Uy[i * ny + j - 1]);
                vh_mid_yt[i * ny + j] = 0.5 * (vh[i * ny + j] + vh[i * ny + j - 1])
                                     - 0.5 * dt / dy *  (Vy[i * ny + j] - Vy[i * ny + j - 1]);
            }
        }
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                Ux_mid_xt[i * ny + j] = uh_mid_xt[i * ny + j] * uh_mid_xt[i * ny + j] / h_mid_xt[i * ny + j]
                                      + 0.5 * g * h_mid_xt[i * ny + j] * h_mid_xt[i * ny + j];
                Uy_mid_yt[i * ny + j] = uh_mid_yt[i * ny + j] * vh_mid_yt[i * ny + j] / h_mid_yt[i * ny + j];
                Vx_mid_xt[i * ny + j] = uh_mid_xt[i * ny + j] * vh_mid_xt[i * ny + j] / h_mid_xt[i * ny + j];
                Vy_mid_yt[i * ny + j] = vh_mid_yt[i * ny + j] * vh_mid_yt[i * ny + j] / h_mid_yt[i * ny + j]
                                      + 0.5 * g * h_mid_yt[i * ny + j] * h_mid_yt[i * ny + j];
            }
        }

        // Now use the mid-point values to predict the values at the next timestep
        for (int i = 1; i < nx -1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                h_new[i * ny + j] = h[i * ny + j] - (dt/dx) * (uh_mid_xt[(i + 1) * ny + j] - uh_mid_xt[i * ny + j])
                                  - (dt/dy) * (vh_mid_yt[i * ny + j + 1] - vh_mid_yt[i * ny + j]);
                uh_new[i * ny + j] = uh[i * ny + j] - (dt/dx) * (Ux_mid_xt[(i + 1) * ny + j] - Ux_mid_xt[i* ny + j])
                                   - (dt/dy) * (Uy_mid_yt[i * ny + j + 1] - Uy_mid_yt[i * ny + j])
                                   + dt * u_accel[i * ny + j] * 0.5 * (h[i * ny + j] + h_new[i * ny + j]);
                vh_new[i * ny + j] = vh[i * ny + j] - (dt/dx) * (Vx_mid_xt[(i + 1) * ny + j] - Vx_mid_xt[i* ny + j])
                                   - (dt/dy) * (Vy_mid_yt[i * ny + j + 1] - Vy_mid_yt[i * ny + j])
                                   + dt * v_accel[i * ny + j] * 0.5 * (h[i * ny + j] + h_new[i * ny + j]);
                u_new[i * ny + j] = uh_new[i * ny + j] / h_new[i * ny + j];
                v_new[i * ny + j] = vh_new[i * ny + j] / h_new[i * ny + j];
            }
        }
    }

    // update inner values u from u_new, v from v_new 
    void update_inner_values() {
        for (int i = 1; i < nx -1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                u[i * ny + j] = u_new[i * ny + j];
                v[i * ny + j] = v_new[i * ny + j];
                h[i * ny + j] = h_new[i * ny + j];
            }
        }
   
    }

    void update_boudary() {
        // first x-slice
        for (int j = 1; j < ny -1; ++j) {
            u[j] = u_new[(nx - 2) * ny + j];
            v[j] = v_new[(nx - 2) * ny + j];
            h[j] = h_new[(nx - 2) * ny + j];
        }
        u[0] = u_new[(nx - 2) * ny + 1];
        v[0] = v_new[(nx - 2) * ny + 1];
        u[ny - 1] = u_new[(nx - 2) * ny -2];
        v[ny - 1] = v_new[(nx - 2) * ny -2];
        // last x-slice
        for (int j = 1; j < ny -1; ++j) {
            u[(nx - 1) * ny + j] = u_new[2 * ny + j];
            v[(nx - 1) * ny + j] = v_new[2 * ny + j];
            h[(nx - 1) * ny + j] = h_new[2 * ny + j];
        }
        u[(nx - 1) * ny] = u_new[2 * ny + 1];
        v[(nx - 1) * ny] = v_new[2 * ny + 1];
        u[(nx - 1) * ny + ny - 1] = u_new[2 * ny + ny - 2];
        v[(nx - 1) * ny + ny - 1] = v_new[2 * ny + ny - 2];
        // no flux from north / south
        for (int i = 0; i < nx; ++i) {
            v[i * ny] = 0.0;
            v[i * ny + ny -1] = 0.0;
        }
        
    }

    void store_snapshot(int i_save, int n) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                u_save[i_save * nx * ny + i * ny + j] = u[i * ny + j];
                v_save[i_save * nx * ny + i * ny + j] = v[i * ny + j];
                h_save[i_save * nx * ny + i * ny + j] = h[i * ny + j];
            }
        }
        t_save[i_save] = n * dt;
    }

    // find the maximunm windspeed in this timestep
    double max_u(double *u, double *v) {
        double max_windspeed_squared = 0.0;
        double temp_windspeed_squared;
        for (int i = 0; i < nx *ny; ++i) {
            temp_windspeed_squared = u[i] * u[i] + v[i] * v[i];
            if (temp_windspeed_squared > max_windspeed_squared) {
                max_windspeed_squared = temp_windspeed_squared;
            }
        }
        double max_windspeed = std::sqrt(max_windspeed_squared);
        return max_windspeed;
    }

    void peek_matrix(double *data) {
        for (int i = 2 * ny; i < 3 * ny; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << " \n";
    }

    // destructor
    ~Grid() {
        delete[] H;
        delete[] height;
        delete[] X;
        delete[] Y;
        delete[] F;
        delete[] u;
        delete[] v;
        delete[] h;
        delete[] u_accel;
        delete[] v_accel;
        delete[] uh;
        delete[] vh;
        delete[] Ux;
        delete[] Uy;
        delete[] Vx;
        delete[] Vy;
        delete[] h_mid_xt;
        delete[] h_mid_yt;
        delete[] uh_mid_xt;
        delete[] uh_mid_yt;
        delete[] vh_mid_xt;
        delete[] vh_mid_yt;
        
        delete[] Ux_mid_xt;
        delete[] Uy_mid_yt;
        delete[] Vx_mid_xt;
        delete[] Vy_mid_yt;

        delete[] h_new;
        delete[] uh_new;
        delete[] vh_new;
        delete[] u_new;
        delete[] v_new;

        delete[] u_save;
        delete[] v_save;
        delete[] h_save;
        delete[] t_save;
    }
};