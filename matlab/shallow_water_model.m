% SHALLOW WATER MODEL
% Copyright (c) 2014 by Robin Hogan
%
% Copying and distribution of this file, with or without modification,
% are permitted in any medium without royalty provided the copyright
% notice and this notice are preserved.  This file is offered as-is,
% without any warranty.
%
% This model integrates the shallow water equations in conservative form
% in a channel using the Lax-Wendroff method.  It can be used to
% illustrate a number of meteorological phenomena.

% ------------------------------------------------------------------
% SECTION 0: Definitions (normally don't modify this section)

% Possible initial conditions of the height field
UNIFORM_WESTERLY=1;
ZONAL_JET=2;
REANALYSIS=3;
GAUSSIAN_BLOB=4;
STEP=5;
CYCLONE_IN_WESTERLY=6;
SHARP_SHEAR=7;
EQUATORIAL_EASTERLY=8;
SINUSOIDAL=9;

% Possible orographies
FLAT=0;
SLOPE=1;
GAUSSIAN_MOUNTAIN=2;
EARTH_OROGRAPHY=3;
SEA_MOUNT=4;

% ------------------------------------------------------------------
% SECTION 1: Configuration
g    = 9.81;                % Acceleration due to gravity (m/s2)
f    = 1e-4;              % Coriolis parameter (s-1)
% f=0.;
beta = 1.6e-11;             % Meridional gradient of f (s-1m-1)
% beta=0.;
% beta=5e-10;

dt_mins              = 1;   % Timestep (minutes)
output_interval_mins = 60;  % Time between outputs (minutes)
forecast_length_days = 4;   % Total simulation length (days)

orography = FLAT;
initial_conditions = GAUSSIAN_BLOB;
initially_geostrophic = false;   % Can be "true" or "false"
add_random_height_noise = false; % Can be "true" or "false"

% If you change the number of gridpoints then orography=EARTH_OROGRAPHY
% or initial_conditions=REANALYSIS won't work
nx=254; % Number of zonal gridpoints
ny=50;  % Number of meridional gridpoints

dx=100.0e3; % Zonal grid spacing (m)
dy=dx;      % Meridional grid spacing

% Specify the range of heights to plot in metres
plot_height_range = [9500 10500];

% ------------------------------------------------------------------
% SECTION 2: Act on the configuration information
dt = dt_mins*60.0; % Timestep (s)
output_interval = output_interval_mins*60.0; % Time between outputs (s)
forecast_length = forecast_length_days*24.0*3600.0; % Forecast length (s)
nt = fix(forecast_length/dt)+1; % Number of timesteps
timesteps_between_outputs = fix(output_interval/dt);
noutput = ceil(nt/timesteps_between_outputs); % Number of output frames

x=(0:nx-1).*dx; % Zonal distance coordinate (m)
y=(0:ny-1).*dy; % Meridional distance coordinate (m)
[Y,X] = meshgrid(y,x); % Create matrices of the coordinate variables

% Create the orography field "H"
switch orography
  case FLAT
    H = zeros(nx, ny);
  case SLOPE
    H = 9000.*2.*abs((mean(x)-X)./max(x));
  case GAUSSIAN_MOUNTAIN
    std_mountain_x = 5.*dx; % Std. dev. of mountain in x direction (m)
    std_mountain_y = 5.*dy; % Std. dev. of mountain in y direction (m)
    H = 4000.*exp(-0.5.*((X-mean(x))./std_mountain_x).^2 ...
                  -0.5.*((Y-mean(y))./std_mountain_y).^2);
  case SEA_MOUNT
    std_mountain = 40.0.*dy; % Standard deviation of mountain (m)
    H = 9250.*exp(-((X-mean(x)).^2+(Y-0.5.*mean(y)).^2)./(2*std_mountain^2));
  case EARTH_OROGRAPHY
    load digital_elevation_map.mat
    H = elevation;
    % Enforce periodic boundary conditions in x
    H([1 end],:)=H([end-1 2],:);
 otherwise
   error(['Don''t know what to do with orography=' num2str(orography)]); 
end

% Create the initial height field 
switch initial_conditions
  case UNIFORM_WESTERLY
    mean_wind_speed = 20; % m/s
    height = 10000-(mean_wind_speed*f/g).*(Y-mean(y)); 
  case SINUSOIDAL
    height = 10000-350.*cos(Y./max(y).*4.*pi);
  case EQUATORIAL_EASTERLY
    height = 10000 - 50.*cos((Y-mean(y)).*4.*pi./max(y));
  case ZONAL_JET
    height = 10000 - tanh(20.0.*((Y-mean(y))./max(y))).*400;
  case REANALYSIS
    load reanalysis.mat
    height = 0.99.*pressure./g;
 case GAUSSIAN_BLOB
   std_blob = 8.0.*dy; % Standard deviation of blob (m)
   height = 9750 + 1000.*exp(-((X-0.25.*mean(x)).^2+(Y-mean(y)).^2)./(2* ...
                                                     std_blob^2));
 case STEP
  height = 9750.*ones(nx, ny);
  height(find(X<max(x)./5 & Y>max(y)/10 & Y<max(y).*0.9)) = 10500;
 case CYCLONE_IN_WESTERLY
   mean_wind_speed = 20; % m/s
   std_blob = 7.0.*dy; % Standard deviation of blob (m)
    height = 10000-(mean_wind_speed*f/g).*(Y-mean(y)) ...
       - 500.*exp(-((X-0.5.*mean(x)).^2+(Y-mean(y)).^2)./(2*std_blob^2));
    max_wind_speed = 20; % m/s
    height = 10250-(max_wind_speed*f/g).*(Y-mean(y)).^2./max(y) ...
       - 1000.*exp(-(0.25.*(X-1.5.*mean(x)).^2+(Y-0.5.*mean(y)).^2)./(2*std_blob^2));
  case SHARP_SHEAR
    mean_wind_speed = 50; % m/s
    height = (mean_wind_speed*f/g).*abs(Y-mean(y));
    height = 10000+height-mean(height(:));
otherwise
   error(['Don''t know what to do with initial_conditions=' num2str(initial_conditions)]); 
end


% Coriolis parameter as a matrix of values varying in y only
F = f+beta.*(Y-mean(y));

% Initialize the wind to rest
u=zeros(nx, ny);
v=zeros(nx, ny);

% We may need to add small-amplitude random noise in order to initialize 
% instability
if add_random_height_noise
  height = height + 1.0.*randn(size(height)).*(dx./1.0e5).*(abs(F)./1e-4);
end


if initially_geostrophic
   % Centred spatial differences to compute geostrophic wind
   u(:,2:end-1) = -(0.5.*g./(F(:,2:end-1).*dx)) ...
       .* (height(:,3:end)-height(:,1:end-2));
   v(2:end-1,:) = (0.5.*g./(F(2:end-1,:).*dx)) ...
       .* (height(3:end,:)-height(1:end-2,:));
   % Zonal wind is periodic so set u(1) and u(end) as dummy points that
   % replicate u(end-1) and u(2), respectively
   u([1 end],:) = u([2 end-1],:);
   % Meridional wind must be zero at the north and south edges of the
   % channel 
   v(:,[1 end]) = 0;

   % Don't allow the initial wind speed to exceed 200 m/s anywhere
   max_wind = 200;
   u(find(u>max_wind)) = max_wind;
   u(find(u<-max_wind)) = -max_wind;
   v(find(v>max_wind)) = max_wind;
   v(find(v<-max_wind)) = -max_wind;
end
% Define h as the depth of the fluid (whereas "height" is the height of
% the upper surface)
h = height - H;

% Initialize the 3D arrays where the output data will be stored
u_save = zeros(nx, ny, noutput);
v_save = zeros(nx, ny, noutput);
h_save = zeros(nx, ny, noutput);
t_save = zeros(1, noutput);

% Index to stored data
i_save = 1;
% ------------------------------------------------------------------
% SECTION 3: Main loop
for n = 1:nt
  % Every fixed number of timesteps we store the fields
  if mod(n-1,timesteps_between_outputs) == 0
    max_u = sqrt(max(u(:).*u(:)+v(:).*v(:)));
    disp(['Time = ' num2str((n-1)*dt/3600) ...
	  ' hours (max ' num2str(forecast_length_days*24) ...
		   '); max(|u|) = ' num2str(max_u)]);
    u_save(:,:,i_save) = u;
    v_save(:,:,i_save) = v;
    h_save(:,:,i_save) = h;
    t_save(i_save) = (n-1).*dt;
    i_save = i_save+1;
  end

  % Compute the accelerations
  u_accel = F(2:end-1,2:end-1).*v(2:end-1,2:end-1) ...
              - (g/(2*dx)).*(H(3:end,2:end-1)-H(1:end-2,2:end-1));
  v_accel = -F(2:end-1,2:end-1).*u(2:end-1,2:end-1) ...
              - (g/(2*dy)).*(H(2:end-1,3:end)-H(2:end-1,1:end-2));

  % Call the Lax-Wendroff scheme to move forward one timestep
  [unew, vnew, h_new] = lax_wendroff(dx, dy, dt, g, u, v, h, ...
                                     u_accel, v_accel);

  % Update the wind and height fields, taking care to enforce 
  % boundary conditions 
  u = unew([end 1:end 1],[1 1:end end]);
  v = vnew([end 1:end 1],[1 1:end end]);
  v(:,[1 end]) = 0;
  h(:,2:end-1) = h_new([end 1:end 1],:);

end

disp('Now run "animate" to animate the simulation');