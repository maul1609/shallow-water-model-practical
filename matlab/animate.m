% This script animates the height field and the vorticity produced by
% a shallow water model. It should be called only after shallow_water_model
% has been run.

% Set the size of the figure
set(gcf,'units','inches');
pos=get(gcf,'position');
pos([3 4]) = [10.5 5];
set(gcf,'position',pos)

% Set other figure properties and draw now
set(gcf,'defaultaxesfontsize',12,...
    'paperpositionmode','auto','color','w');
drawnow

% Axis units are thousands of kilometers (x and y are in metres)
x_1000km = x.*1e-6;
y_1000km = y.*1e-6;

% Set colormap to have 64 entries
ncol=64;
colormap(jet(ncol));
% colormap(hclmultseq01);
% colormap(flipud(cbrewer('div','RdBu',32)))

% Interval between arrows in the velocity vector plot
interval = 6;

% Set this to "true" to save each frame as a png file
plot_frames = false;

% Decide whether to show height in metres or km
if mean(plot_height_range) > 1000
  height_scale = 0.001;
  height_title = 'Height (km)';
else
  height_scale = 1;
  height_title = 'Height (m)';
end

disp(['Maximum orography height = ' num2str(max(H(:))) ' m']);

% Loop through the frames of the animation
for it = 1:noutput
  clf

  % Extract the height and velocity components for this frame
  h = squeeze(h_save(:,:,it));
  u = squeeze(u_save(:,:,it));
  v = squeeze(v_save(:,:,it));

  % First plot the height field
  subplot(2,1,1);

  % Plot the height field
  handle = image(x_1000km, y_1000km, (h'+H').*height_scale);
  set(handle,'CDataMapping','scaled');
  set(gca,'ydir','normal');
  caxis(plot_height_range.*height_scale);

  % Plot the orography as black contours every 1000 m
  hold on
  warning off
  contour(x_1000km, y_1000km, H',[1:1000:8001],'k');
  warning on

  % Plot the velocity vectors
  quiver(x_1000km(3:interval:end), y_1000km(3:interval:end), ...
         u(3:interval:end, 3:interval:end)',...
         v(3:interval:end, 3:interval:end)','k','linewidth',0.2);

  % Write the axis labels, title and time of the frame
  xlabel('X distance (1000s of km)');
  ylabel('Y distance (1000s of km)');
  title(['\bf' height_title]);
  text(0, max(y_1000km), ['Time = ' num2str(t_save(it)./3600) ' hours'],...
       'verticalalignment','bottom','fontsize',12);

  % Set other axes properties and plot a colorbar
  daspect([1 1 1]);
  axis([0 max(x_1000km) 0 max(y_1000km)]);
  colorbar
  
  % Compute the vorticity
  vorticity = zeros(size(u));
  vorticity(2:end-1,2:end-1) = (1/dy).*(u(2:end-1,1:end-2)-u(2:end-1,3:end)) ...
     + (1/dx).*(v(3:end,2:end-1)-v(1:end-2,2:end-1));

  % Now plot the vorticity
  subplot(2,1,2);
  handle = image(x_1000km, y_1000km, vorticity');
  set(handle,'CDataMapping','scaled');
  set(gca,'ydir','normal');
  caxis([-3 3].*1e-4);

  % Axis labels and title
  xlabel('X distance (1000s of km)');
  ylabel('Y distance (1000s of km)');
  title('\bfRelative vorticity (s^{-1})');

  % Other axes properties and plot a colorbar
  daspect([1 1 1]);
  axis([0 max(x_1000km) 0 max(y_1000km)]);
  colorbar

  % Now render this frame
  warning off
  drawnow
  warning on

  % To make an animation we can save the frames as a 
  % sequence of images
  if plot_frames
      eval(['print -dpng frame',num2str(it,'%03d'),'.png']);
%     imwrite(frame2im(getframe(gcf)),...
% 	    ['frame' num2str(it,'%03d') '.png']);
  end
   pause(0.05);
end
