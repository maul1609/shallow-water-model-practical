def lax_wendroff(dx, dy, dt, g, u, v, h, u_tendency, v_tendency):

   # This function performs one timestep of the Lax-Wendroff scheme
   # applied to the shallow water equations

   # First work out mid-point values in time and space
   uh = u*h;
   vh = v*h;

   h_mid_xt = 0.5*(h[1:,:]+h[0:-1,:]) \
      -(0.5*dt/dx)*(uh[1:,:]-uh[0:-1,:]);
   h_mid_yt = 0.5*(h[:,1:]+h[:,0:-1]) \
      -(0.5*dt/dy)*(vh[:,1:]-vh[:,0:-1]);

   Ux = uh*u+0.5*g*h**2.;
   Uy = uh*v;
   uh_mid_xt = 0.5*(uh[1:,:]+uh[0:-1,:]) \
      -(0.5*dt/dx)*(Ux[1:,:]-Ux[0:-1,:]);
   uh_mid_yt = 0.5*(uh[:,1:]+uh[:,0:-1]) \
      -(0.5*dt/dy)*(Uy[:,1:]-Uy[:,0:-1]);

   Vx = Uy;
   Vy = vh*v+0.5*g*h**2.;
   vh_mid_xt = 0.5*(vh[1:,:]+vh[0:-1,:]) \
      -(0.5*dt/dx)*(Vx[1:,:]-Vx[0:-1,:]);
   vh_mid_yt = 0.5*(vh[:,1:]+vh[:,0:-1]) \
      -(0.5*dt/dy)*(Vy[:,1:]-Vy[:,0:-1]);

   # Now use the mid-point values to predict the values at the next
   # timestep
   h_new = h[1:-1,1:-1] \
      - (dt/dx)*(uh_mid_xt[1:,1:-1]-uh_mid_xt[0:-1,1:-1]) \
      - (dt/dy)*(vh_mid_yt[1:-1,1:]-vh_mid_yt[1:-1,0:-1]);


   Ux_mid_xt = uh_mid_xt*uh_mid_xt/h_mid_xt + 0.5*g*h_mid_xt**2.;
   Uy_mid_yt = uh_mid_yt*vh_mid_yt/h_mid_yt;
   uh_new = uh[1:-1,1:-1] \
      - (dt/dx)*(Ux_mid_xt[1:,1:-1]-Ux_mid_xt[0:-1,1:-1]) \
      - (dt/dy)*(Uy_mid_yt[1:-1,1:]-Uy_mid_yt[1:-1,0:-1]) \
      + dt*u_tendency*0.5*(h[1:-1,1:-1]+h_new);


   Vx_mid_xt = uh_mid_xt*vh_mid_xt/h_mid_xt;
   Vy_mid_yt = vh_mid_yt*vh_mid_yt/h_mid_yt + 0.5*g*h_mid_yt**2.;
   vh_new = vh[1:-1,1:-1] \
      - (dt/dx)*(Vx_mid_xt[1:,1:-1]-Vx_mid_xt[0:-1,1:-1]) \
      - (dt/dy)*(Vy_mid_yt[1:-1,1:]-Vy_mid_yt[1:-1,0:-1]) \
      + dt*v_tendency*0.5*(h[1:-1,1:-1]+h_new);
   u_new = uh_new/h_new;
   v_new = vh_new/h_new;
   
   return (u_new, v_new, h_new)
