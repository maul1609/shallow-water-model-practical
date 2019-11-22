# This script animates the height field and the vorticity produced by
# a shallow water model. It should be called only after shallow_water_model
# has been run.

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)



#f,(ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
f=plt.figure()
ax1=f.add_subplot(211)
ax2=f.add_subplot(212)

ax1.autoscale(enable=True, axis='y', tight=True)

# Axis units are thousands of kilometers (x and y are in metres)
x_1000km = x * 1.e-6
y_1000km = y * 1.e-6

# Set colormap to have 64 entries
ncol=64;

# Interval between arrows in the velocity vector plot
interval = 6;

# Set this to "True" to save each frame as a png file
plot_frames = False;

# Decide whether to show height in metres or km
if np.mean(plot_height_range) > 1000:
   height_scale = 0.001;
   height_title = 'Height (km)';
else:
   height_scale = 1;
   height_title = 'Height (m)';


print('Maximum orography height = %f m' % np.max(H[:]));
u = np.squeeze(u_save[:,:,0]);
vorticity = np.zeros(np.shape(u));

# Loop through the frames of the animation
for it in range(0,noutput):

   # Extract the height and velocity components for this frame
   h = np.squeeze(h_save[:,:,it]);
   u = np.squeeze(u_save[:,:,it]);
   v = np.squeeze(v_save[:,:,it]);

   # Compute the vorticity
   vorticity[1:-1,1:-1] = (1./dy)*(u[1:-1,0:-2]-u[1:-1,2:]) \
     + (1./dx)*(v[2:,1:-1]-v[0:-2,1:-1]);
   # First plot the height field

   if it==0:
    
      # Plot the height field
      im=ax1.imshow(np.transpose(h+H)*height_scale, \
        extent=[np.min(x_1000km),np.max(x_1000km),np.min(y_1000km),np.max(y_1000km)], \
        cmap='jet',origin='lower')
      # Set other axes properties and plot a colorbar
      cb1=f.colorbar(im,ax=ax1)
      cb1.set_label('height (km)')
      # Contour the terrain:
      cs=ax1.contour(x_1000km,y_1000km,np.transpose(H),levels=range(1,11001,1000),colors='k')
      
      # Plot the velocity vectors
      Q = ax1.quiver(x_1000km[2::interval],y_1000km[2::interval], \
         np.transpose(u[2::interval,2::interval]), \
         np.transpose(v[2::interval,2::interval]), scale=5e2,scale_units='xy',pivot='mid')
      ax1.set_ylabel('Y distance (1000s of km)');
      ax1.set_title(height_title);
      tx1=ax1.text(0, np.max(y_1000km), 'Time = %.1f hours' % (t_save[it]/3600.));
      
      
      # Now plot the vorticity
      im2=ax2.imshow(np.transpose(vorticity), \
        extent=[np.min(x_1000km),np.max(x_1000km),np.min(y_1000km),np.max(y_1000km)], \
        cmap='jet',origin='lower')
      # Set other axes properties and plot a colorbar
      cb2=f.colorbar(im2,ax=ax2)
      cb2.set_label('vorticity (s$^{-1}$)')
      ax2.set_xlabel('X distance (1000s of km)');
      ax2.set_ylabel('Y distance (1000s of km)');
      ax2.set_title('Relative vorticity (s$^{-1}$)');
      tx2=ax2.text(0, np.max(y_1000km), 'Time = %.1f hours' % (t_save[it]/3600.));
      
   else:
      # top plot:
      im.set_data(np.transpose(H+h)*height_scale)
      cs.set_array(np.transpose(h))
      Q.set_UVC(np.transpose(u[2::interval,2::interval]), \
               np.transpose(v[2::interval,2::interval]))
      tx1.set_text('Time = %.1f hours' % (t_save[it]/3600.));
      
      # bottom plot:
      im2.set_data(np.transpose(vorticity))
      tx2.set_text('Time = %.1f hours' % (t_save[it]/3600.));
   
   
      
   im.set_clim((plot_height_range*height_scale));
   im2.set_clim((-3e-4,3e-4));
   ax1.axis((0., np.max(x_1000km), 0., np.max(y_1000km)));
   ax2.axis((0., np.max(x_1000km), 0., np.max(y_1000km)));

   
   


  
   # To make an animation we can save the frames as a 
   # sequence of images
   if plot_frames:
      plt.savefig('frame%03d.png' % it,format='png') 

   plt.pause(0.05)
