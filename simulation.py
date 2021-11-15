import os, sys

from numpy.core.fromnumeric import repeat
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.ffmpeg_path'] = r'D:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
import numpy as np

matplotlib.rcParams.update({'figure.facecolor': 'white'})
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

from numba import njit


def animate_hist(history):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    xdata, ydata = [], []

    Nx = history[0].shape[0]
    Ny = history[0].shape[1]

    X, Y = np.meshgrid( np.arange(0, Nx), np.arange(0, Ny) )

    wire = ax.plot_wireframe(X, Y, history[0], rcount=25, ccount=25, linewidth=0.6)

    ax.set_zlim(-1, 1)
    ax.set_xlim(0, Nx-1)
    ax.set_ylim(0, Ny-1)

    def update(frame):
        ax.clear()
        ax.set_zlim(-1, 1)
        ax.set_xlim(0, Nx-1)
        ax.set_ylim(0, Ny-1)
        wire = ax.plot_wireframe(X, Y, history[frame], rcount=25, ccount=25, linewidth=0.6)
        print(len(history)-frame)
        return wire,

    ani = FuncAnimation(fig, update, frames=np.arange(0, history.shape[0]), blit=False)
    ani.save('./animation.mp4', writer='ffmpeg', dpi=100, fps=30)
    # plt.show()

    return None


@njit
def force(field, ind):
    k = 1.0
    f_sum = - k * field[ind[0], ind[1], 0]
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            indx = (ind[0] + i) % field.shape[0]
            indy = (ind[1] + j) % field.shape[1]
            # if indx > 0 and indy > 0 and indx < field.shape[0] and indy < field.shape[1]:
            f_sum += - k * ( field[ind[0], ind[1], 0] - field[indx, indy, 0])
    return f_sum

@njit
def simulate(field, dt, iterations):
    m = 1.0
    for e in range(iterations):
        force_field = np.zeros((field.shape[0], field.shape[1]))
        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                force_field[x, y] = force(field, (x, y))
        field[:, :, 1] += force_field / m * dt
        field[:, :, 0] += field[:, :, 1] * dt
    return field

if __name__ == "__main__": 

    dt = 0.001
    Nx = 101
    Ny = 101

    x = np.arange(0, Nx, 1, float)
    y = x[:,np.newaxis]
    X, Y = np.meshgrid(x, y)
    x0 = int(Nx/2)
    y0 = int(Ny/2)

    fwhm = Nx / 10
    scale = 0.4

    field = np.zeros((Nx, Ny, 2))
    field[:, :, 0] = scale * np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)
    ax.set_xlim(0, Nx-1)
    ax.set_ylim(0, Ny-1)
    ax.plot_wireframe(X, Y, field[:, :, 0])
    # plt.show()

    sim_len = 100000
    step_size = 200

    field_hist = np.zeros((int(sim_len/step_size), Nx, Ny))

    for e in range(0, int(sim_len/step_size), 1):

        field = simulate(field, dt, step_size)
        pos = field[:, :, 0]
        vel = field[:, :, 1]
        field_hist[e] = np.copy(pos)

    animate_hist(field_hist)
