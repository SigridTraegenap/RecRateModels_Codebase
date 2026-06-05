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





def generate_bar_center(Size, center_x, center_y,
                        wid, ang,length=None,
                 space_resolution=1):

    """
    Size (int): Number of columns in the grid.
    pos (float): position of the bar: 0-1.
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

    # Convert angle to radians for trigonometric functions
    angle_rad = np.deg2rad(ang)

    # Calculate direction vector components for the angle
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)

    #define bar
    if length is None:
        length = np.sqrt(2)*Size
    half_length = length / 2
    half_width =  wid / 2

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

def generate_bar(Size, pos, angle,
                 bar_width, bar_length=None,
                 dx=1,
                 pbc=True):

    center_x, center_y = get_center(Size, pos, angle,
                                    space_resolution=dx)

    bar = generate_bar_center(Size, center_x, center_y,
                            bar_width, ang=90-angle,
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
                                bar_width, ang=90-angle,
                                length=bar_length,
                         space_resolution=1)
        bar += bar2


    return bar

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
    return ellipse


def get_center(Size, pos,  ang,
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
        return Size *np.sqrt(2)* (
            np.cos(np.deg2rad(45)-angle_rad))

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

if __name__=='__main__':
    import matplotlib.patches as patches
    N=50
    for ang in [0,30,90,120,150,180,210,240,270,300,330,360]:
    # for ang in [0,5,10]:


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

        ##test perpendicular line
        orth_vec = np.asarray([pos0[1], -pos0[0]])/np.linalg.norm(pos0)
        start = pos0.copy()
        end1 = start + 30*orth_vec
        end2 = start - 30*orth_vec
        orth_line=np.stack([end2, start, end1])
        plt.plot(orth_line[:,0], orth_line[:,1], color="k")


        pos0=get_center(N, 1, ang,
                        space_resolution=1)
        plt.scatter(pos0[0], pos0[1], color='red')

        pos0=get_center(N, 0.5, ang,
                        space_resolution=1)
        plt.scatter(pos0[0], pos0[1], color='orange')


        #plt.axis([0,N,0,N])
        ax=plt.gca()
        ax.set_aspect('equal', 'box')

        #Create a Rectangle patch
        rect = patches.Rectangle((-N/2,-N/2), N, N, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()




    # #%%
    # N=50
    # ang=30
    # wid=3

    # for pos in [0,0.25,0.5,0.75,1]:
    #     center_x, center_y = get_center(N, pos, ang)
    #     bar = generate_bar_center(N, center_x, center_y,
    #                             wid, ang=90-ang,length=None,
    #                      space_resolution=1)

    #     plt.imshow(bar, origin='lower')
    #     plt.scatter(center_x+N/2, center_y+N/2)
    #     plt.show()


    # #%%
    # N=50
    # ang=280
    # wid=3

    # for pos in [-0.5,-0.25,0,0.25,0.5,0.75,1]:

    #     bar = generateEllipse(N, pos, ang,
    #                      2, np.inf,
    #                      dx=1,
    #                      pbc=True)

    #     plt.imshow(bar, origin='lower')

    #     center_x, center_y = get_center(N, pos, ang)
    #     plt.scatter(center_x+N/2, center_y+N/2)


    #     if pos<0.5:
    #         pos_new=pos+1

    #     if pos>=0.5:
    #         pos_new=pos-1

    #     center_x, center_y = get_center(N, pos_new, ang)
    #     plt.scatter(center_x+N/2, center_y+N/2, s=1)


    #     plt.show()

    #%%
    def fx(alpha):
        # if alpha>=45 and alpha<=90 or (alpha>=225 and ang<=270):
        #     alpha = 90-alpha

        return np.sqrt(2)*np.cos(np.deg2rad(45-alpha))

    alpha=np.linspace(0,90,100)
    Ls = fx(alpha)
    plt.plot(alpha, Ls)
