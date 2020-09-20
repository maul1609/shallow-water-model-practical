#include "hip/hip_runtime.h"
#include <iostream>
#include <string>
#define double float

__global__ void compute_accleration(double *u_accel, double *v_accel, double *F, double *u, double *v, double g, double dx, double dy, double *H, int nx, int ny);

__global__ void lax_wendroff_step1(int nx, int ny, double *uh, double *u, double *h, double *vh, double *v, double *Ux, double g, double *Uy, double *Vx, double *Vy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= 0 && i < nx) {
        if (j >= 0 && j < ny) {
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
}

__global__ void lax_wendroff_step2(int nx, int ny, double *uh, double *u, double *h, double *vh, double *v, double *Ux, double g, double *Uy, double *Vx, double *Vy, double *h_mid_xt, double *uh_mid_xt, double *vh_mid_xt, double *h_mid_yt, double *uh_mid_yt, double *vh_mid_yt, double *Ux_mid_xt, double *Uy_mid_yt, double *Vx_mid_xt, double *Vy_mid_yt, double dx, double dy, double dt) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= 1 && i < nx) {
        if (j >= 0 && j < ny) {
            h_mid_xt[i * ny + j] = 0.5 * (h[i * ny + j] + h[(i - 1) * ny + j]) \
                                   - 0.5 * dt / dx *  (uh[i * ny + j] - uh[(i - 1) * ny + j]);
            uh_mid_xt[i * ny + j] = 0.5 * (uh[i * ny + j] + uh[(i - 1) * ny + j])
                - 0.5 * dt / dx *  (Ux[i * ny + j] - Ux[(i - 1) * ny + j]);
            vh_mid_xt[i * ny + j] = 0.5 * (vh[i * ny + j] + vh[(i - 1) * ny + j])
                - 0.5 * dt / dx *  (Vx[i * ny + j] - Vx[(i - 1) * ny + j]);
        }
    }
    if (i >= 0 && i < nx) {
        if (j >= 1 && j < ny) {
            h_mid_yt[i * ny + j] = 0.5 * (h[i * ny + j] + h[i * ny + j - 1])
                - 0.5 * dt / dy *  (vh[i * ny + j] - vh[i * ny + j - 1]);
            uh_mid_yt[i * ny + j] = 0.5 * (uh[i * ny + j] + uh[i * ny + j - 1])
                - 0.5 * dt / dy *  (Uy[i * ny + j] - Uy[i * ny + j - 1]);
            vh_mid_yt[i * ny + j] = 0.5 * (vh[i * ny + j] + vh[i * ny + j - 1])
                - 0.5 * dt / dy *  (Vy[i * ny + j] - Vy[i * ny + j - 1]);
        }
    }
    if (i >= 0 && i < nx) {
        if (j >= 0 && j < ny) {
            Ux_mid_xt[i * ny + j] = uh_mid_xt[i * ny + j] * uh_mid_xt[i * ny + j] / h_mid_xt[i * ny + j]
                + 0.5 * g * h_mid_xt[i * ny + j] * h_mid_xt[i * ny + j];
            Uy_mid_yt[i * ny + j] = uh_mid_yt[i * ny + j] * vh_mid_yt[i * ny + j] / h_mid_yt[i * ny + j];
            Vx_mid_xt[i * ny + j] = uh_mid_xt[i * ny + j] * vh_mid_xt[i * ny + j] / h_mid_xt[i * ny + j];
            Vy_mid_yt[i * ny + j] = vh_mid_yt[i * ny + j] * vh_mid_yt[i * ny + j] / h_mid_yt[i * ny + j]
                + 0.5 * g * h_mid_yt[i * ny + j] * h_mid_yt[i * ny + j];
        }
    }
}

__global__ void lax_wendroff_step3(int nx, int ny, double *u, double *v, double *h, double *u_accel, double *v_accel, double *uh, double *vh, double *h_mid_xt, double *uh_mid_xt, double *vh_mid_xt, double *h_mid_yt, double *uh_mid_yt, double *vh_mid_yt, double *Ux_mid_xt, double *Uy_mid_yt, double *Vx_mid_xt, double *Vy_mid_yt, double dx, double dy, double dt, double *h_new, double *u_new, double *v_new) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= 1 && i < nx) {
        if (j >= 0 && j < ny) {
            h_new[i * ny + j] = h[i * ny + j] - (dt/dx) * (uh_mid_xt[(i + 1) * ny + j] - uh_mid_xt[i * ny + j])
                - (dt/dy) * (vh_mid_yt[i * ny + j + 1] - vh_mid_yt[i * ny + j]);
            double uh_new = uh[i * ny + j] - (dt/dx) * (Ux_mid_xt[(i + 1) * ny + j] - Ux_mid_xt[i* ny + j])
                - (dt/dy) * (Uy_mid_yt[i * ny + j + 1] - Uy_mid_yt[i * ny + j])
                + dt * u_accel[i * ny + j] * 0.5 * (h[i * ny + j] + h_new[i * ny + j]);
            double vh_new = vh[i * ny + j] - (dt/dx) * (Vx_mid_xt[(i + 1) * ny + j] - Vx_mid_xt[i* ny + j])
                - (dt/dy) * (Vy_mid_yt[i * ny + j + 1] - Vy_mid_yt[i * ny + j])
                + dt * v_accel[i * ny + j] * 0.5 * (h[i * ny + j] + h_new[i * ny + j]);
            u_new[i * ny + j] = uh_new / h_new[i * ny + j];
            v_new[i * ny + j] = vh_new / h_new[i * ny + j];
        }
    }
    // update inner values
    if (i >= 1 && i < nx-1) {
        if (j >= 1 && j < ny-1) {
            u[i * ny + j] = u_new[i * ny + j];
            v[i * ny + j] = v_new[i * ny + j];
            h[i * ny + j] = h_new[i * ny + j];
        }
    }
}

__global__ void update_boudary(int nx, int ny, double *u, double *v, double *h, double *h_new, double *u_new, double *v_new) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    // first x-slice
    if (i == 0) {
        if (j >= 1 && j < ny -1) {
            u[j] = u_new[(nx - 2) * ny + j];
            v[j] = v_new[(nx - 2) * ny + j];
            h[j] = h_new[(nx - 2) * ny + j];
            u[(nx - 1) * ny + j] = u_new[2 * ny + j];
            v[(nx - 1) * ny + j] = v_new[2 * ny + j];
            h[(nx - 1) * ny + j] = h_new[2 * ny + j];

        }
        if (j == 0) {
            u[0] = u_new[(nx - 2) * ny + 1];
            v[0] = v_new[(nx - 2) * ny + 1];
            u[ny - 1] = u_new[(nx - 2) * ny -2];
            v[ny - 1] = v_new[(nx - 2) * ny -2];
            u[(nx - 1) * ny] = u_new[2 * ny + 1];
            v[(nx - 1) * ny] = v_new[2 * ny + 1];
            u[(nx - 1) * ny + ny - 1] = u_new[2 * ny + ny - 2];
            v[(nx - 1) * ny + ny - 1] = v_new[2 * ny + ny - 2];

        }

    }
    // no flux from north / south
    if (j == 0) {
        if (i < nx) {
            v[i * ny] = 0.0;
            v[i * ny + ny -1] = 0.0;
        }
    }
}
__global__ void store_data(int nx, int ny, double *u, double *v, double *h, double *u_save, double *v_save, double *h_save, int index) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    int displacement = nx * ny * index;
    if (i >= 0 && i < nx) {
        if (j >= 0 && j < ny) {
            u_save[displacement + i * ny + j] = u[i * ny + j];
            v_save[displacement + i * ny + j] = v[i * ny + j];
            h_save[displacement + i * ny + j] = h[i * ny + j];
        }
    }

}

void calculate_max_windspeed(double *u_save, double *v_save, int nx, int ny, int noutputs) {
    double temp_windspeed_squared;
    for (int n = 0; n < noutputs; ++n) {
        double max_windspeed_squared = 0.0;
        for (int i = n*nx *ny; i < (n+1)* nx *ny; ++i) {
            temp_windspeed_squared = u_save[i] * u_save[i] + v_save[i] * v_save[i];
            if (temp_windspeed_squared > max_windspeed_squared) {
                max_windspeed_squared = temp_windspeed_squared;
            }
        }
        double max_windspeed = std::sqrt(max_windspeed_squared);
        std::cout << "Time = " << n << ", max windspeed = " << max_windspeed << "\n";
    }
}

int main(int agrc, char *argv[]) {
    std::string initial_conditions = "UNIFORM_WESTERLY";
    std::string orographies = "FLAT";

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



    double *X_cpu; // matrices of the coordinate variables
    double *Y_cpu; // matrices of the coordinate variables
    double *H_cpu; // the orography field "H"
    double *height_cpu; // the height of the upper surface)
    double *pressure_cpu;
    double *F_cpu; // Coriolis parameter
    double *u_cpu, *v_cpu; // wind speed matrices in x and y direction
    double *h_cpu; // Define h as the depth of the fluid

    double *u_save_cpu, *v_save_cpu, *h_save_cpu, *t_save_cpu;

    // allocate memory in the cpu
    H_cpu = new double[nx * ny];
    height_cpu = new double[nx * ny];
    pressure_cpu = new double[nx * ny];
    X_cpu = new double[nx * ny];
    Y_cpu = new double[nx * ny];
    F_cpu = new double[nx * ny];
    u_cpu = new double[nx * ny];
    v_cpu = new double[nx * ny];
    h_cpu = new double[nx * ny];

    // initial_conditions == UNIFORM_WESTERLY
    double mean_wind_speed = 20.0;
    double meanY = (ny - 1) * dy / 2.0;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            // Create X, Y coordinate matrices
            X_cpu[i * ny + j] = i * dx;
            Y_cpu[i * ny + j] = j * dy;
            // orography == FLAT
            H_cpu[i * ny + j] = 0.0;
            // initial_conditions == UNIFORM_WESTERLY
            height_cpu[i * ny + j] = 10000.0 - (mean_wind_speed * f / g) * (Y_cpu[i * ny + j] - meanY);
            // Coriolis parameter as a matrix of values varying in y only
            F_cpu[i * ny + j] = f + beta * (Y_cpu[i * ny + j] - meanY);
            // Initialize the wind to rest
            u_cpu[i * ny + j] = 0.0;
            v_cpu[i * ny + j] = 0.0;
            // Define h as the depth of the fluid (whereas "height" is the height of the upper surface)
            h_cpu[i * ny + j] = height_cpu[i * ny + j] - H_cpu[i * ny + j];

        }

    }

    //create_3d_arrays
    u_save_cpu = new double[nx * ny * noutput];
    v_save_cpu = new double[nx * ny * noutput];
    h_save_cpu = new double[nx * ny * noutput];
    t_save_cpu = new double[noutput];

    //allocate memory in the gpu
    double *H; // the orography field "H"
    double *F; // Coriolis parameter
    double *u, *v; // wind speed matrices in x and y direction
    double *h; // Define h as the depth of the fluid

    double *u_save, *v_save, *h_save, *t_save;
    double *u_accel, *v_accel;

    double *uh, *vh, *Ux, *Uy, *Vx, *Vy;
    double *h_mid_xt, *h_mid_yt, *uh_mid_xt, *uh_mid_yt, *vh_mid_xt, *vh_mid_yt;
    double *Ux_mid_xt, *Uy_mid_yt, *Vx_mid_xt, *Vy_mid_yt;
    double *h_new, *uh_new, *vh_new, *u_new, *v_new;

    hipMalloc ((void **) &H, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &F, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &u, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &v, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &h, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &u_save, sizeof(double)*(nx*ny*noutput));
    hipMalloc ((void **) &v_save, sizeof(double)*(nx*ny*noutput));
    hipMalloc ((void **) &h_save, sizeof(double)*(nx*ny*noutput));
    hipMalloc ((void **) &u_accel, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &v_accel, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &uh, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &vh, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Ux, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Uy, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Vx, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Vy, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &h_mid_xt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &h_mid_yt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &uh_mid_xt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &uh_mid_yt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &vh_mid_xt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &vh_mid_yt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Ux_mid_xt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Uy_mid_yt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Vx_mid_xt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &Vy_mid_yt, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &h_new, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &uh_new, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &vh_new, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &u_new, sizeof(double)*(nx*ny));
    hipMalloc ((void **) &v_new, sizeof(double)*(nx*ny));

    // copy data from cpu to gpu
    hipMemcpy(H, H_cpu, sizeof(double)*(nx*ny), hipMemcpyHostToDevice);
    hipMemcpy(F, F_cpu, sizeof(double)*(nx*ny), hipMemcpyHostToDevice);
    hipMemcpy(u, u_cpu, sizeof(double)*(nx*ny), hipMemcpyHostToDevice);
    hipMemcpy(v, v_cpu, sizeof(double)*(nx*ny), hipMemcpyHostToDevice);
    hipMemcpy(h, h_cpu, sizeof(double)*(nx*ny), hipMemcpyHostToDevice);

    int block_size = 32;
    dim3 dimBlock(block_size);
    dim3 dimGrid(nx/block_size+1 , ny);

    int i_save = 0;
    for (int n = 0; n < nt; ++n) {
        if (n % timesteps_between_outputs == 0) {
            //store data in index
            int index = n / timesteps_between_outputs;
            hipLaunchKernelGGL(store_data, dim3(dimGrid), dim3(dimBlock), 0, 0, nx, ny, u, v, h, u_save, v_save, h_save, index);
            ++i_save;
        }
        hipDeviceSynchronize();
        hipLaunchKernelGGL(compute_accleration, dim3(dimGrid), dim3(dimBlock), 0, 0, u_accel, v_accel, F, u, v, g, dx, dy, H, nx, ny);
        hipLaunchKernelGGL(lax_wendroff_step1, dim3(dimGrid), dim3(dimBlock), 0, 0, nx, ny, uh, u, h, vh, v, Ux, g, Uy, Vx, Vy);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(lax_wendroff_step2, dim3(dimGrid), dim3(dimBlock), 0, 0, nx, ny, uh, u, h, vh, v, Ux, g, Uy, Vx, Vy, h_mid_xt, uh_mid_xt, vh_mid_xt, h_mid_yt, uh_mid_yt, vh_mid_yt, Ux_mid_xt, Uy_mid_yt, Vx_mid_xt, Vy_mid_yt, dx, dy, dt);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(lax_wendroff_step3, dim3(dimGrid), dim3(dimBlock), 0, 0, nx, ny, u, v, h, u_accel, v_accel, uh, vh, h_mid_xt, uh_mid_xt, vh_mid_xt, h_mid_yt, uh_mid_yt, vh_mid_yt, Ux_mid_xt, Uy_mid_yt, Vx_mid_xt, Vy_mid_yt,dx, dy, dt, h_new, u_new, v_new);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(update_boudary, dim3(dimGrid), dim3(dimBlock), 0, 0, nx, ny, u, v, h, h_new, u_new, v_new);
        hipDeviceSynchronize();
    }
    // copy data from gpu to cpu
    hipMemcpy(u_save_cpu, u_save, sizeof(double)*(nx*ny*noutput), hipMemcpyDeviceToHost);
    hipMemcpy(v_save_cpu, v_save, sizeof(double)*(nx*ny*noutput), hipMemcpyDeviceToHost);
    hipMemcpy(h_save_cpu, h_save, sizeof(double)*(nx*ny*noutput), hipMemcpyDeviceToHost);
    calculate_max_windspeed(u_save_cpu, v_save_cpu, nx, ny, noutput);

    // free cpu memory
    delete[] H_cpu;
    delete[] height_cpu;
    delete[] pressure_cpu;
    delete[] X_cpu;
    delete[] Y_cpu;
    delete[] F_cpu;
    delete[] u_cpu;
    delete[] v_cpu;
    delete[] h_cpu;

    //free gpu memory
    hipFree(H);
    hipFree(F);
    hipFree(u);
    hipFree(v);
    hipFree(h);
    hipFree(u_save);
    hipFree(v_save);
    hipFree(h_save);
    hipFree(u_accel);
    hipFree(v_accel);
    hipFree(uh);
    hipFree(vh);
    hipFree(Ux);
    hipFree(Uy);
    hipFree(Vx);
    hipFree(Vy);
    hipFree(h_mid_xt);
    hipFree(h_mid_yt);
    hipFree(uh_mid_xt);
    hipFree(uh_mid_yt);
    hipFree(vh_mid_xt);
    hipFree(vh_mid_yt);
    hipFree(Ux_mid_xt);
    hipFree(Uy_mid_yt);
    hipFree(Vx_mid_xt);
    hipFree(Vy_mid_yt);
    hipFree(h_new);
    hipFree(uh_new);
    hipFree(vh_new);
    hipFree(u_new);
    hipFree(v_new);
    
    return 0; 
}

__global__ void compute_accleration(double *u_accel, double *v_accel, double *F, double *u, double *v, double g, double dx, double dy, double *H, int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= 1 && i < nx -1) {
        if (j >= 1 && j < ny -1) {
            u_accel[i * ny + j] = F[i * ny + j] * v[i * ny + j]
                - (g / (2.0 * dx)) * (H[(i + 1) * ny + j] - H[(i - 1) * ny + j]);
            v_accel[i * ny + j] = -F[i * ny + j] * u[i * ny + j]
                - (g / (2.0 * dy)) * (H[i * ny + j + 1] - H[i * ny + j - 1]);
        }

    }
}

