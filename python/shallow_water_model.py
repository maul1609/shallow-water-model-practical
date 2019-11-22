# SHALLOW WATER MODEL
# Copyright (c) 2017 by Paul Connolly
#
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved.  This file is offered as-is,
# without any warranty.
#
# This model integrates the shallow water equations in conservative form
# in a channel using the Lax-Wendroff method.  It can be used to
# illustrate a number of meteorological phenomena.

# ------------------------------------------------------------------
import numpy as np
import sys
from scipy.special import erfcinv as erfcinv
import tqdm as tqdm
import time
import scipy.io as sio
import lax_wendroff as lw

# SECTION 0: Definitions (normally don't modify this section)

# Possible initial conditions of the height field
UNIFORM_WESTERLY=1;
ZONAL_JET=2;
REANALYSIS=3;
GAUSSIAN_BLOB=4;
STEP=5;
CYCLONE_IN_WESTERLY=6;
SHARP_SHEAR=7;
EQUATORIAL_EASTERLY=8;
SINUSOIDAL=9;

# Possible orographies
FLAT=0;
SLOPE=1;
GAUSSIAN_MOUNTAIN=2;
EARTH_OROGRAPHY=3;
SEA_MOUNT=4;

# ------------------------------------------------------------------
# SECTION 1: Configuration
g    = 9.81;                # Acceleration due to gravity (m/s2)
f    = 1e-4;              # Coriolis parameter (s-1)
#f=0.;
beta = 1.6e-11;             # Meridional gradient of f (s-1m-1)
#beta=0.;
#beta=5e-10;

dt_mins              = 1.;   # Timestep (minutes)
output_interval_mins = 60.;  # Time between outputs (minutes)
forecast_length_days = 4.;   # Total simulation length (days)

orography = FLAT
initial_conditions = GAUSSIAN_BLOB;
initially_geostrophic = False;   # Can be "True" or "False"
add_random_height_noise = False; # Can be "True" or "False"

# If you change the number of gridpoints then orography=EARTH_OROGRAPHY
# or initial_conditions=REANALYSIS won't work
nx=254; # Number of zonal gridpoints
ny=50;  # Number of meridional gridpoints

dx=100.0e3; # Zonal grid spacing (m)
dy=dx;      # Meridional grid spacing

# Specify the range of heights to plot in metres
plot_height_range = np.array([9500., 10500.]);

# ------------------------------------------------------------------
# SECTION 2: Act on the configuration information
dt = dt_mins*60.0; # Timestep (s)
output_interval = output_interval_mins*60.0; # Time between outputs (s)
forecast_length = forecast_length_days*24.0*3600.0; # Forecast length (s)
nt = int(np.fix(forecast_length/dt)+1); # Number of timesteps
timesteps_between_outputs = np.fix(output_interval/dt);
noutput = int(np.ceil(nt/timesteps_between_outputs)); # Number of output frames

x=np.mgrid[0:nx]*dx; # Zonal distance coordinate (m)
y=np.mgrid[0:ny]*dy; # Meridional distance coordinate (m)
[Y,X] = np.meshgrid(y,x); # Create matrices of the coordinate variables



# Create the orography field "H"
if orography == FLAT:
   H = np.zeros((nx, ny));
elif orography == SLOPE:
   H = 9000.*2.*np.abs((np.mean(x)-X)/np.max(x));
elif orography == GAUSSIAN_MOUNTAIN:
   std_mountain_x = 5.*dx; # Std. dev. of mountain in x direction (m)
   std_mountain_y = 5.*dy; # Std. dev. of mountain in y direction (m)
   H = 4000.*np.exp(-0.5*((X-np.mean(x))/std_mountain_x)**2. \
                  -0.5*((Y-np.mean(y))/std_mountain_y)**2.);
elif orography == SEA_MOUNT:
   std_mountain = 40.0*dy; # Standard deviation of mountain (m)
   H = 9250.*np.exp(-((X-np.mean(x))**2.+(Y-0.5*np.mean(y))**2.)/(2.*std_mountain**2.));
elif orography == EARTH_OROGRAPHY:
   mat_contents = sio.loadmat('digital_elevation_map.mat')
   H = mat_contents['elevation'];
   # Enforce periodic boundary conditions in x
   H[[0, -1],:]=H[[-2, 1],:];
else:
   print('Don''t know what to do with orography=' + np.num2str(orography)); 
   sys.exit()



# Create the initial height field 
if initial_conditions == UNIFORM_WESTERLY:
   mean_wind_speed = 20.; # m/s
   height = 10000.-(mean_wind_speed*f/g)*(Y-np.mean(y)); 
elif initial_conditions == SINUSOIDAL:
   height = 10000.-350.*np.cos(Y/np.max(y)*4.*np.pi);
elif initial_conditions == EQUATORIAL_EASTERLY:
   height = 10000. - 50.*np.cos((Y-np.mean(y))*4.*np.pi/np.max(y));
elif initial_conditions == ZONAL_JET:
   height = 10000. - np.tanh(20.0*((Y-np.mean(y))/np.max(y)))*400.;
elif initial_conditions == REANALYSIS:
   mat_contents = sio.loadmat('reanalysis.mat')
   pressure = mat_contents['pressure'];
   height = 0.99*pressure/g;
elif initial_conditions == GAUSSIAN_BLOB:
   std_blob = 8.0*dy; # Standard deviation of blob (m)
   height = 9750. + 1000.*np.exp(-((X-0.25*np.mean(x))**2.+(Y-np.mean(y))**2.)/(2.* \
                                                     std_blob**2.));
elif initial_conditions == STEP:
   height = 9750.*np.ones((nx, ny));
   height[where((X<np.max(x)/5.) & (Y>np.max(y)/10.) & (Y<np.max(y)*0.9))] = 10500.;
elif initial_conditions == CYCLONE_IN_WESTERLY:
   mean_wind_speed = 20.; # m/s
   std_blob = 7.0*dy; # Standard deviation of blob (m)
   height = 10000.-(mean_wind_speed*f/g)*(Y-np.mean(y)) \
       - 500.*np.exp(-((X-0.5*np.mean(x))**2.+(Y-np.mean(y))**2.)/(2.*std_blob**2.));
   max_wind_speed = 20.; # m/s
   height = 10250.-(max_wind_speed*f/g)*(Y-np.mean(y))**2./np.max(y) \
       - 1000.*np.exp(-(0.25*(X-1.5*np.mean(x))**2.+(Y-0.5*np.mean(y))**2.)/(2.*std_blob**2.));
elif initial_conditions == SHARP_SHEAR:
   mean_wind_speed = 50.; # m/s
   height = (mean_wind_speed*f/g)*np.abs(Y-np.mean(y));
   height = 10000.+height-np.mean(height[:]);
else:
   print("Don't know what to do with initial_conditions=%f" % initial_conditions); 
   sys.exit()


# Coriolis parameter as a matrix of values varying in y only
F = f+beta*(Y-np.mean(y));

# Initialize the wind to rest
u=np.zeros((nx, ny));
v=np.zeros((nx, ny));

# We may need to add small-amplitude random noise in order to initialize 
# instability
if add_random_height_noise:
   r,c=np.shape(height)
   height = height + 1.0*np.random.randn(r,c)*(dx/1.0e5)*(np.abs(F)/1e-4);


if initially_geostrophic:

   # Centred spatial differences to compute geostrophic wind
   u[:,1:-1] = -(0.5*g/(F[:,1:-1]*dx)) \
       * (height[:,2:]-height[:,0:-2]);
   v[1:-1,:] = (0.5*g/(F[1:-1,:]*dx)) \
       * (height[2:,:]-height[0:-2,:]);
   # Zonal wind is periodic so set u(1) and u(end) as dummy points that
   # replicate u(end-1) and u(2), respectively
   u[[0 ,-1],:] = u[[1 ,-2],:];
   # Meridional wind must be zero at the north and south edges of the
   # channel 
   v[:,[0, -1]] = 0.;

   # Don't allow the initial wind speed to exceed 200 m/s anywhere
   max_wind = 200.;
   u[np.where(u>max_wind)] = max_wind;
   u[np.where(u<-max_wind)] = -max_wind;
   v[np.where(v>max_wind)] = max_wind;
   v[np.where(v<-max_wind)] = -max_wind;


# Define h as the depth of the fluid (whereas "height" is the height of
# the upper surface)
h = height - H;

# Initialize the 3D arrays where the output data will be stored
u_save = np.zeros((nx, ny, noutput));
v_save = np.zeros((nx, ny, noutput));
h_save = np.zeros((nx, ny, noutput));
t_save = np.zeros((noutput,1));

# Index to stored data
i_save = 0;

# ------------------------------------------------------------------
# SECTION 3: Main loop
for n in range(0,nt):
   # Every fixed number of timesteps we store the fields
   if np.mod(n,timesteps_between_outputs) == 0:
   
      max_u = np.sqrt(np.max(u[:]*u[:]+v[:]*v[:]));
      
      print("Time = %f hours (max %f); max(|u|) = %f"  
          % ((n)*dt/3600., forecast_length_days*24., max_u) )
		   
      u_save[:,:,i_save] = u;
      v_save[:,:,i_save] = v;
      h_save[:,:,i_save] = h;
      t_save[i_save] = (n)*dt;
      i_save = i_save+1;
  

   # Compute the accelerations
   u_accel = F[1:-1,1:-1]*v[1:-1,1:-1] \
              - (g/(2.*dx))*(H[2:,1:-1]-H[0:-2,1:-1]);
   v_accel = -F[1:-1,1:-1]*u[1:-1,1:-1] \
              - (g/(2.*dy))*(H[1:-1,2:]-H[1:-1,0:-2]);

   # Call the Lax-Wendroff scheme to move forward one timestep
   (unew, vnew, h_new) = lw.lax_wendroff(dx, dy, dt, g, u, v, h, u_accel, v_accel);

   # Update the wind and height fields, taking care to enforce 
   # boundary conditions    
   u[1:-1,1:-1] = unew;
   v[1:-1,1:-1] = vnew;
   
   # first x-slice
   u[0,1:-1]=unew[-1,:]
   u[0,0]=unew[-1,0]
   u[0,-1]=unew[-1,-1]
   v[0,1:-1]=vnew[-1,:]
   v[0,0]=vnew[-1,0]
   v[0,-1]=vnew[-1,-1]
   # last x-slice
   u[-1,1:-1]=unew[1,:]
   u[-1,0]=unew[1,0]
   u[-1,-1]=unew[1,-1]
   v[-1,1:-1]=vnew[1,:]
   v[-1,0]=vnew[1,0]
   v[-1,-1]=vnew[1,-1]
   
   # no flux from north / south
   v[:,[0,-1]]=0.;
   # interior
   h[1:-1,1:-1] = h_new;
   # first x-slice
   h[0,1:-1]=h_new[-1,:]
   # last x-slice
   h[-1,1:-1]=h_new[1,:]
   
   

print('Now run "animate" to animate the simulation');

