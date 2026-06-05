#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:07:02 2024

@author: sigridtragenap
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.transform import rotate
from tools_general.filter_funcs import get_additivenoise_ideal


def SinuisoidGrating(Size, A=1, angle_deg=0,
                     frequency_Kspace=10,
                     phase=0, Pos=[0,0],
                     dx=1):
    phase = np.asarray(phase)
    # Generate Sinusoid grating
    #Size
    x = np.arange(-Size/2.0,Size/2.,dx)
    X, Y = np.meshgrid(x, x)
    X, Y = X/Size * np.pi*2, Y/Size * np.pi*2

    angle = np.deg2rad(angle_deg)
    if len(phase.shape)>0:
        grating=[]

        for phi in phase:
            grating.append(A*np.cos(
               frequency_Kspace*X*np.cos(angle) +
               frequency_Kspace*Y*np.sin(angle) -
               phi))
        grating = np.asarray(grating)

    if len(phase.shape)==0:
        grating = A*np.cos(
           frequency_Kspace*X*np.cos(angle) +
           frequency_Kspace*Y*np.sin(angle) -
           phase)
    return grating


def generate_bar_old(Size, pos, wid, ang,
                 space_resolution=1):

    """
    M (int): Number of rows in the grid.
    N (int): Number of columns in the grid.
    pos (float): position of the bar: 0-1.
    wid (int): Width of the bar.
    ang (float): Angle of the bar in degree.

    Returns:
        numpy.ndarray: MxN array with the bar drawn at specified position and angle.
    """

    N = int(Size/space_resolution)
    M = int(Size/space_resolution)

    # Initialize the grid with a black background
    grid = np.zeros((M, N), dtype=float)

    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(ang)

    # Calculate direction vector components for the angle
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    perp_dx, perp_dy = -dy, dx  # Perpendicular vector

    # Define the length to ensure the bar spans beyond the grid diagonally
    if dx <= 0.2:
        max_length = M
    else:
        max_length = int(N/dx)  #  diagonal length

    # Calculate the bar's center point based on `pos`
    length = max_length
    center_y = M / 2 + (pos - 0.5) * length * perp_dy
    center_x = N / 2 + (pos - 0.5) * length * perp_dx

    # Calculate the four corners of the rectangle around the center point
    half_length = max_length / 2
    half_width = wid / 2

    # Corner displacements before rotation
    corners = [
        (-half_length, -half_width), (half_length, -half_width),
        (half_length, half_width), (-half_length, half_width)
    ]

    # Rotate corners and translate to the center position
    rotated_corners = [
        (
            int(center_y + corner[0] * dy - corner[1] * dx),
            int(center_x + corner[0] * dx + corner[1] * dy)
        )
        for corner in corners
    ]
    #print('pos, ang',pos,ang,'corners',rotated_corners)

    # Extract row and column coordinates for the rectangle
    rr, cc = polygon(
        [c[0] for c in rotated_corners],
        [c[1] for c in rotated_corners],
        shape=grid.shape
    )

    # Fill the rectangle in the grid
    grid[rr, cc] = 1  # Set bar values to maximum intensity

    return grid

def get_center(Size, pos, ang, space_resolution=1):
    """
    Calculates the center of the bar for a given position phase (0 to 1) and angle.
    Reversed trajectory to match the original right-to-left / top-to-bottom flow.
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(-ang)

    # The baseline distance the bar travels across the grid
    length_trajectory = Size#*1.4
    #* (np.abs(np.cos(ang)) + np.abs(np.sin(ang)))

    # Use (0.5 - pos) instead of (pos - 0.5) to invert the travel direction
    xpos = (0.5 - pos) * length_trajectory * np.cos(angle_rad)
    ypos = (0.5 - pos) * length_trajectory * np.sin(angle_rad)

    return np.asarray([xpos, ypos])

def get_center_old(Size, pos,  ang,
               space_resolution=1):

    """
    M (int): Number of rows in the grid.
    N (int): Number of columns in the grid.
    pos (float): position of the bar: 0-1.
    ang (float): Angle of the bar in degree.

    Returns:
        numpy.ndarray: MxN array with the bar drawn at specified position and angle.
    """

    def calc_length_tracjectory(angle_rad, Size):
        return Size
    #np.abs(Size * np.sqrt(2)*np.cos(np.pi/4-angle_rad))

    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(ang)
    #Size * (1/np.sin(angle_rad) + np.cos(angle_rad)/np.tan(angle_rad))

    #calculate length of trajectory
    if (ang>=0 and ang<=45) or (ang>=180 and ang<=225):
        length_trajectory = calc_length_tracjectory(angle_rad, Size)

        if ang>=180:
            length_trajectory *=-1
        xpos = (pos-0.5)*length_trajectory * np.cos(np.pi-angle_rad)
        ypos = (pos-0.5)*length_trajectory * np.sin(np.pi-angle_rad)


    if ang>=45 and ang<=90 or (ang>=225 and ang<=270):
        angle_rad = np.deg2rad(90-ang)
        length_trajectory = length_trajectory = calc_length_tracjectory(angle_rad, Size)

        if ang>=180:
            length_trajectory *=-1

        ypos = (pos-0.5)*length_trajectory * np.cos(angle_rad)
        xpos = -1 * (pos-0.5)*length_trajectory * np.sin(angle_rad)

    if ang>=90 and ang<=135 or (ang>=270 and ang<=315):
        angle_rad = np.deg2rad(90+ang)
        length_trajectory = length_trajectory = calc_length_tracjectory(angle_rad, Size)

        if ang>=180:
            length_trajectory *=-1

        ypos = (pos-0.5)*length_trajectory * np.cos(angle_rad)
        xpos = (pos-0.5)*length_trajectory * np.sin(angle_rad)

    if ang>=135 and ang<=180 or (ang>=315 and ang<=360):
        angle_rad = np.deg2rad(180-ang)
        length_trajectory = length_trajectory = calc_length_tracjectory(angle_rad, Size)


        if ang>180:
            length_trajectory *=-1

        xpos = -1*(pos-0.5)*length_trajectory * np.cos(np.pi-angle_rad)
        ypos = (pos-0.5)*length_trajectory * np.sin(np.pi-angle_rad)


    return np.asarray([xpos, ypos])

def generate_bar_center(Size, center_x, center_y,
                        wid, ang,length=None,
                 space_resolution=1):

    """
    Size (int): Number of columns in the grid.
    wid (int): Width of the bar.
    ang (float): Angle of the bar in degree.

    Returns:
        numpy.ndarray: MxN array with the bar drawn at specified position and angle.
    """

    center_y = Size/2 + center_y
    center_x = Size/2 + center_x

    N = int(Size/space_resolution)
    M = int(Size/space_resolution)

    # Initialize the grid with a black background
    grid = np.zeros((M, N), dtype=float)

    #define bar
    if length is None:
        length = np.sqrt(2)*Size
    half_length = length / 2
    half_width =  wid / 2

    # Corner displacements before rotation
    corners = [
        (-half_width, -half_length), (half_width, -half_length),
        (half_width, half_length), (-half_width, half_length)
    ]


    #Define Affine transform
    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(ang)

    # Calculate direction vector components for the angle
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    #Rotation matrix
    R_angle = np.asarray([[dx,dy], [-dy,dx]])
    #translation vector
    translation_shift = np.asarray([center_x,center_y])

    corners_vec = np.asarray(corners)
    corners_rotated = (R_angle@corners_vec.T).T

    corners_final = corners_rotated + translation_shift

    rr,cc =polygon(corners_final[:,0],corners_final[:,1], shape=(Size,Size))
    grid= np.zeros((Size,Size), dtype=float)
    grid[rr,cc]=1

    return grid

def generate_bar(Size, pos, angle,
                 bar_width, bar_length=None,
                 dx=1,
                 pbc=True):

    center_x, center_y = get_center(Size, pos, angle,
                                    space_resolution=dx)

    bar = generate_bar_center(Size, center_x, center_y,
                            bar_width, ang=angle,
                            length=bar_length,
                     space_resolution=1)

    if pbc:
        if pos<0.5:
            pos_new=pos+1

        if pos>=0.5:
            pos_new=pos-1
        center_x, center_y = get_center(Size, pos_new, angle,
                                        space_resolution=dx)

        bar2 = generate_bar_center(Size, center_x, center_y,
                                bar_width, ang=angle,
                                length=bar_length,
                         space_resolution=1)
        bar += bar2


    return bar

def generate_bumpy_bar(Size, pos, angle, bar_width, noise_params,
                       bar_length=None, dx=1, pbc=True):
    # Get the center for the primary bar position
    center_x, center_y = get_center(Size, pos, angle, space_resolution=dx)

    # Generate the primary bumpy bar
    bar = generate_bumpy_bar_center(Size, center_x, center_y,
                                    bar_width, ang=angle,
                                    length=bar_length,
                                    noise_params=noise_params,
                                    space_resolution=1)

    if pbc:
        # Calculate the wrapped position for periodicity
        pos_new = pos + 1 if pos < 0.5 else pos - 1

        center_x_wrap, center_y_wrap = get_center(Size, pos_new, angle, space_resolution=dx)

        # Generate the second part of the bar for the wrap-around
        bar2 = generate_bumpy_bar_center(Size, center_x_wrap, center_y_wrap,
                                         bar_width, ang=angle,
                                         length=bar_length,
                                         noise_params=noise_params,
                                         space_resolution=1)
        bar += bar2

    return bar

def generate_bumpy_bar_center(Size, center_x, center_y, wid, ang,
                              noise_params, length=None, space_resolution=1):
    # Re-use your existing geometric logic to define the bar mask
    center_y_abs = Size/2 + center_y
    center_x_abs = Size/2 + center_x

    if length is None:
        length = np.sqrt(2)*Size

    half_length, half_width = length / 2, wid / 2
    corners = [(-half_width, -half_length), (half_width, -half_length),
               (half_width, half_length), (-half_width, half_length)]

    angle_rad = np.deg2rad(ang)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    R_angle = np.asarray([[dx, dy], [-dy, dx]])

    corners_rotated = (R_angle @ np.asarray(corners).T).T
    corners_final = corners_rotated + np.asarray([center_x_abs, center_y_abs])

    # Create the binary mask for the bar
    rr, cc = polygon(corners_final[:, 0], corners_final[:, 1], shape=(Size, Size))
    bar_mask = np.zeros((Size, Size), dtype=float)
    bar_mask[rr, cc] = 1

    # Generate and apply noise
    # We use (1, Size, Size) because your function expects a full_shape including events
    noise_raw = get_additivenoise_ideal(full_shape=(1, Size, Size),
                                        seed=noise_params.get('seed', 81622),
                                        dist_btw_peaks_min=noise_params['dist_min'],
                                        dist_btw_peaks_max=noise_params['dist_max'])[0]

    # Normalize noise to zero-mean, scale by std, and apply only to the bar area
    noise_norm = (noise_raw - np.mean(noise_raw)) / np.std(noise_raw)
    bumpy_bar = bar_mask + (noise_norm * noise_params.get('noise_std', 0.2) * bar_mask)

    # Rectify to prevent negative rates
    bumpy_bar = np.clip(bumpy_bar, a_min=0, a_max=None)

    # Normalize: Ensure the total drive (sum) is identical to the original bar_mask
    if np.sum(bumpy_bar) > 0:
        bumpy_bar *= (np.sum(bar_mask) / np.sum(bumpy_bar))

    return bumpy_bar

def getEllipsoid(Size, Pos=(0,0), angle_deg=0,
                 sigmas=(10,10), dx=1):

    angle = np.deg2rad(angle_deg)
    x = np.arange(-Size/2.0,Size/2.,dx)
    X, Y = np.meshgrid(x, x)

    a = np.cos(angle)**2/(2*sigmas[0])+np.sin(angle)**2/(2*sigmas[1])
    b = -np.sin(2*angle)/(4*sigmas[0])+np.sin(2*angle)/(4*sigmas[1])
    c = np.sin(angle)**2/(2*sigmas[0])+np.cos(angle)**2/(2*sigmas[1])

    gauss = np.exp(- (a*(X-Pos[0])**2 +
                      c*(Y-Pos[1])**2 +
                    2*b*(X-Pos[0])*(Y-Pos[1])
                     )
                   )
    return gauss #/np.abs(gauss).sum()

def generateEllipse(Size, pos, angle,
                 sigma1, sigma2,
                 dx=1,
                 pbc=True):

    center_x, center_y = get_center(Size, pos, angle,
                                    space_resolution=dx)

    ellipse = getEllipsoid(Size, Pos=(center_x, center_y),
                             angle_deg=angle,
                            sigmas=(sigma1, sigma2),
                            dx=dx)

    if pbc:
        if pos<0.5:
            pos_new=pos+1

        if pos>=0.5:
            pos_new=pos-1

        center_x, center_y = get_center(Size, pos_new, angle,
                                        space_resolution=dx)

        ellipse2 = getEllipsoid(Size, Pos=(center_x, center_y),
                                 angle_deg=angle,
                                sigmas=(sigma1, sigma2),
                                dx=dx)
        ellipse += ellipse2


    #ellipse /= np.abs(ellipse).sum()
    return ellipse.T


def generate_curved_bar_center(Size, center_x, center_y, wid, ang, length=None, space_resolution=1, curvedness=0.0):
    """
    Generates a curved bar (crescent) at a specific center point.
    Perfectly aligned with the orientation of the original polygon-based generate_bar_center.
    """
    center_y_abs = Size/2 + center_y
    center_x_abs = Size/2 + center_x

    if length is None:
        length = np.sqrt(2)*Size

    N = int(Size/space_resolution)
    M = int(Size/space_resolution)

    x_coords = np.arange(N) * space_resolution
    y_coords = np.arange(M) * space_resolution

    # 'ij' indexing makes X map to rows and Y map to columns,
    # perfectly mimicking skimage.draw.polygon!
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    dX = X - center_x_abs
    dY = Y - center_y_abs

    # Direction vector components
    rad = np.deg2rad(ang)
    dx_a = np.cos(rad)
    dy_a = np.sin(rad)

    # Exact inverse of the [[dx, dy], [-dy, dx]] rotation matrix used in your old code
    x_local = dX * dx_a - dY * dy_a
    y_local = dX * dy_a + dY * dx_a

    # Apply parabolic deformation: x' = x + c * y^2
    x_deformed = x_local + curvedness * (y_local ** 2)

    # Apply bounds
    in_width = np.abs(x_deformed) <= wid / 2
    in_length = np.abs(y_local) <= length / 2

    grid = (in_width & in_length).astype(float)

    return grid


def generate_curved_bar(Size, pos, angle, bar_width, curvedness=0.0, bar_length=None,
                        meander_amp=0.0, meander_freq=1.0, dx=1, pbc=True):
    """
    Generates a straight or curved bar following a linear or meandering trajectory,
    handling periodic boundaries.
    """
    # 1. Get the primary center (meandering or linear)
    center_x, center_y = get_meandering_center(Size, pos, angle,
                                               meander_amp=meander_amp,
                                               meander_freq=meander_freq,
                                               space_resolution=dx)

    # 2. Draw the primary shape
    bar = generate_curved_bar_center(Size, center_x, center_y,
                                     wid=bar_width, ang=angle,
                                     length=bar_length,
                                     space_resolution=dx,
                                     curvedness=curvedness)

    # 3. Handle Periodic Boundary Conditions
    if pbc:
        if pos < 0.5:
            pos_new = pos + 1
        else:
            pos_new = pos - 1

        # Get the wrapped center
        center_x_wrap, center_y_wrap = get_meandering_center(Size, pos_new, angle,
                                                             meander_amp=meander_amp,
                                                             meander_freq=meander_freq,
                                                             space_resolution=dx)

        # Draw the wrapped portion
        bar_wrap = generate_curved_bar_center(Size, center_x_wrap, center_y_wrap,
                                              wid=bar_width, ang=angle,
                                              length=bar_length,
                                              space_resolution=dx,
                                              curvedness=curvedness)
        bar += bar_wrap

    return np.clip(bar, a_min=0, a_max=1)



def get_meandering_center(Size, pos, ang, meander_amp=0.0, meander_freq=1.0, space_resolution=1):
    """
    Wraps get_center to add a perpendicular sinusoidal wobble.
    Empirically calculates the perpendicular vector to handle the quadrant flips.
    """
    # 1. Get the base linear center
    cx_base, cy_base = get_center(Size, pos, ang, space_resolution)

    # 2. If no meandering, return early
    if meander_amp == 0.0:
        return np.asarray([cx_base, cy_base])

    # 3. Find the TRUE trajectory vector by taking a tiny step forward
    # Using 0.001 to get the instantaneous tangent vector
    cx_next, cy_next = get_center(Size, pos + 0.001, ang, space_resolution)

    traj_dx = cx_next - cx_base
    traj_dy = cy_next - cy_base

    # Normalize the trajectory vector
    norm = np.sqrt(traj_dx**2 + traj_dy**2)
    if norm > 0:
        traj_dx /= norm
        traj_dy /= norm

    # 4. Calculate the true perpendicular vector (rotate 90 degrees)
    perp_dx = -traj_dy
    perp_dy = traj_dx

    # 5. Add sinusoidal displacement
    wobble = meander_amp * np.sin(2 * np.pi * meander_freq * pos)

    return np.asarray([cx_base + wobble * perp_dx, cy_base + wobble * perp_dy])


def generate_bumps(Size, pos, angle, spacing=10, sigma=None,
                   n_rows=1, row_spacing=None, pbc=True):
    """
    A grid of n_rows × n_bumps Gaussian blobs that looks like a dotted bar.

    The number of blobs along the bar axis is derived automatically as
    round(Size / spacing), so the dots tile the full grid at the given
    domain spacing.  For a 60-unit grid with spacing=10 you get 6 bumps.

    Parameters
    ----------
    spacing     : distance between blob centres along the long axis (default 10)
    sigma       : Gaussian radius of each blob (default: spacing / 2.5)
    n_rows      : number of rows of dots across the bar width (1 or 2)
    row_spacing : distance between rows in the width direction
                  (default: sigma * 2)
    pbc         : periodic boundary conditions

    Coordinate note
    ---------------
    get_center returns (cx, cy) where cx → ROW and cy → COLUMN.
    getEllipsoid(Pos=(p0, p1)) treats p0 as COLUMN and p1 as ROW.
    Bar long axis in (row, col): (sin(ang), cos(ang))  →  d_col=cos, d_row=sin.
    Bar width axis in (row, col): (cos(ang), -sin(ang)) →  d_col=-sin, d_row=cos.
    """
    angle_rad = np.deg2rad(angle)

    n_bumps = max(1, round(Size / spacing))
    if sigma is None:
        sigma = spacing / 2.5
    if row_spacing is None:
        row_spacing = sigma * 2.0

    along_offsets = np.arange(n_bumps) - (n_bumps - 1) / 2.0
    across_offsets = np.arange(n_rows)  - (n_rows  - 1) / 2.0

    def _place_at_center(cx, cy):
        """cx = row offset, cy = col offset (from get_center)."""
        grid = np.zeros((Size, Size))
        for ka in along_offsets:
            for kr in across_offsets:
                # displacement along bar long axis
                p_col = cy + ka * spacing     * np.cos(angle_rad)
                p_row = cx + ka * spacing     * np.sin(angle_rad)
                # displacement across bar width
                p_col += kr * row_spacing * (-np.sin(angle_rad))
                p_row += kr * row_spacing *   np.cos(angle_rad)
                grid += getEllipsoid(Size, Pos=(p_col, p_row),
                                     sigmas=(sigma, sigma))
        return grid

    cx, cy = get_center(Size, pos, angle)
    bumps = _place_at_center(cx, cy)

    if pbc:
        pos_wrap = pos + 1 if pos < 0.5 else pos - 1
        cx_w, cy_w = get_center(Size, pos_wrap, angle)
        bumps += _place_at_center(cx_w, cy_w)

    return bumps


if __name__=='__main__':
    import matplotlib.patches as patches
    N=50
    for ang in [0,30,90,120,150,180,210,240,270,300,330,360]:

    # for ang in [0,90,]:


        pos = np.linspace(0,1,20)

        center=[]
        for p in pos:
            center.append(get_center(N, p, ang,
                           space_resolution=1))
        center = np.asarray(center)

        plt.scatter(center[:,0], center[:,1])

        pos0=get_center(N, 0, ang,
                       space_resolution=1)
        plt.scatter(pos0[0], pos0[1], color='green')

        ##test perpendicular line to start
        orth_vec = np.asarray([pos0[1], -pos0[0]])/np.linalg.norm(pos0)
        start = pos0.copy()
        end1 = start + 30*orth_vec
        end2 = start - 30*orth_vec
        orth_line=np.stack([end2, start, end1])
        plt.plot(orth_line[:,0], orth_line[:,1], color="k")

        # pos0=get_center_old(N, 1, ang,
        #                space_resolution=1)
        # plt.scatter(pos0[0], pos0[1], color='pink')


        pos0=get_center(N, 1, ang,
                       space_resolution=1)
        plt.scatter(pos0[0], pos0[1], color='red')

        pos0=get_center(N, 0.5, ang,
                       space_resolution=1)
        plt.scatter(pos0[0], pos0[1], color='orange')


        pos0=get_center_old(N, 1, ang,
                       space_resolution=1)
        plt.scatter(pos0[0], pos0[1], color='pink', s=1)





        #plt.axis([0,N,0,N])
        ax=plt.gca()
        ax.set_aspect('equal', 'box')

        #Create a Rectangle patch
        rect = patches.Rectangle((-N/2,-N/2), N, N, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()




    #%%
    N=50
    ang=30
    wid=3

    for pos in [0,0.25,0.5,0.75,1]:
        center_x, center_y = get_center(N, pos, ang)
        bar = generate_bar_center(N, center_x, center_y,
                                wid, ang=ang,length=None,
                         space_resolution=1)

        plt.imshow(bar.T, origin='lower')
        plt.scatter(center_x+N/2, center_y+N/2)
        plt.show()


    #%%
    N=50
    ang=30
    wid=3

    # for pos in [-0.5,-0.25,0,0.25,0.5,0.75,1]:
    for pos in [0,0.25,0.5,0.75,1]:

        bar = generateEllipse(N, pos, ang,
                         2, np.inf,
                         dx=1,
                         pbc=True)

        plt.imshow(bar.T, origin='lower')

        center_x, center_y = get_center(N, pos, ang)
        plt.scatter(center_x+N/2, center_y+N/2)


        if pos<0.5:
            pos_new=pos+1

        if pos>=0.5:
            pos_new=pos-1

        center_x, center_y = get_center(N, pos_new, ang)
        plt.scatter(center_x+N/2, center_y+N/2, s=1)


        plt.show()

    #%%
    N=50
    ang=270
    wid=20

    # for pos in [-0.5,-0.25,0,0.25,0.5,0.75,1]:
    for pos in [0,0.25,0.5,0.75,1]:

        bar =   generate_bar(N, pos, ang,
                         20, None,
                         dx=1,
                         pbc=True)

        plt.imshow(bar.T, origin='lower')

        center_x, center_y = get_center(N, pos, ang)
        plt.scatter(center_x+N/2, center_y+N/2)


        if pos<0.5:
            pos_new=pos+1

        if pos>=0.5:
            pos_new=pos-1

        center_x, center_y = get_center(N, pos_new, ang)
        plt.scatter(center_x+N/2, center_y+N/2, s=1)


        plt.show()

    #%%est Crescent / Curved Bar
    N = 50
    ang = 30
    wid = 3
    curvedness = 0.01 # This is the parameter that bends it into a crescent!

    for pos in [0, 0.25, 0.5, 0.75, 1]:

        # Generate the curved bar instead of the standard bar
        crescent = generate_curved_bar(
            Size=N,
            pos=pos,
            angle=ang,
            bar_width=wid,
            curvedness=curvedness,
            bar_length=2*N,
            dx=1,
            pbc=False
        )

        plt.imshow(crescent.T, origin='lower', cmap='viridis')
        center_x, center_y = get_center(N, pos, ang)
        plt.scatter(center_x+N/2, center_y+N/2)


        if pos<0.5:
            pos_new=pos+1

        if pos>=0.5:
            pos_new=pos-1

        center_x, center_y = get_center(N, pos_new, ang)
        plt.scatter(center_x+N/2, center_y+N/2, s=1)
        plt.show()

    #%%
    #%% Test Full Curved Bar with PBC
    N = 50
    ang = 90
    wid = 3
    curvedness = 0.03  # Set to >0 to see the crescent bend

    # Sweeping across the grid
    for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:

        # 1. Call the wrapper function with pbc=True
        crescent_with_pbc = generate_curved_bar(
            Size=N,
            pos=pos,
            angle=ang,
            bar_width=wid,
            curvedness=curvedness,
            bar_length=2*N, # Keeping it long so the curve is obvious
            dx=1,
            pbc=True        # <-- Activating Periodic Boundary Conditions
        )

        # 2. Plot the resulting grid
        plt.imshow(crescent_with_pbc.T, origin='lower', cmap='viridis')

        # 3. Calculate and plot the primary center (Red)
        center_x, center_y = get_center(N, pos, ang)
        plt.scatter(center_x + N/2, center_y + N/2, color='red', label='Primary Center')

        # 4. Calculate and plot the wrapped center (Orange)
        if pos < 0.5:
            pos_new = pos + 1
        else:
            pos_new = pos - 1

        center_x_wrap, center_y_wrap = get_center(N, pos_new, ang)
        plt.scatter(center_x_wrap + N/2, center_y_wrap + N/2, color='orange', s=15, label='Wrapped Center')

        plt.title(f"Curved Bar (PBC On) - Phase: {pos}")
        plt.show()


        #%% Test Meandering Trajectory
    N = 50
    ang = 120
    wid = 3
    curvedness = 0.01  # Crescent shape

    # Let's track the trajectory over more phases to see the sine wave clearly
    phases = np.linspace(0, 1, 10)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 1. Linear Path (meander_amp = 0)
    grid_linear = np.zeros((N, N))
    for pos in phases:
        crescent = generate_curved_bar(Size=N, pos=pos, angle=ang, bar_width=wid,
                                       curvedness=curvedness, bar_length=40,
                                       meander_amp=0.0, pbc=True) # AMP = 0
        grid_linear = np.maximum(grid_linear, crescent.T)

    axs[0].imshow(grid_linear, origin='lower', cmap='magma')
    axs[0].set_title("Linear Trajectory (amp=0)")

    # 2. Meandering Path (meander_amp = 10)
    grid_meander = np.zeros((N, N))
    for pos in phases:
        crescent = generate_curved_bar(Size=N, pos=pos, angle=ang, bar_width=wid,
                                       curvedness=curvedness, bar_length=40,
                                       meander_amp=20.0, meander_freq=0.5, pbc=True) # AMP = 10
        grid_meander = np.maximum(grid_meander, crescent.T)

    axs[1].imshow(grid_meander, origin='lower', cmap='magma')
    axs[1].set_title("Meandering Trajectory (amp=10)")

    plt.tight_layout()
    plt.show()
