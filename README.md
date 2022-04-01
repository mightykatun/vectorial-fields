# Python-Vectorial-Fields
# Vectorial fields in fluids simulator.

# modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# variables 1
v_density = 30
l_density = 80
p_density = 80
d_density = 500
c_pos = (1, 1)
c_radius = 0.95

# variables 2
l_viscosity = 148.1408929
rho_a = 7850
rho_g = 1260
radius = 0.0005
pi = np.pi
g = 9.81
sum_f = (
    (
        -
        (rho_g * 2 * (4/3) * pi * radius**3 * g)
        -
        (6 * pi * radius * l_viscosity)
        +
        (rho_a * (4/3) * 2 * pi * radius**3 * g)
    )
)

# main
def flow(
        psi,
        mask = None,
        x=np.linspace(-4, 4, d_density),
        y=np.linspace(-4, 4, d_density),
        h=1e-10
):
    x_axis, y_axis = np.meshgrid(x, y)
    if mask:
        x_axis, y_axis = (
            np.ma.masked_where(mask(x_axis, y_axis), x_axis),
            np.ma.masked_where(mask(x_axis, y_axis), y_axis)
        )

    # complex analysis navier-stokes
    u_axis = - (psi(x_axis, y_axis + h) - psi(x_axis, y_axis - h)) / (2 * h)
    v_axis = (psi(x_axis + h, y_axis) - psi(x_axis - h, y_axis)) / (2 * h)

    plt.figure(figsize=(8, 8), dpi=p_density)
    plt.quiver(
        x_axis[::v_density, ::v_density],
        y_axis[::v_density, ::v_density],
        u_axis[::v_density, ::v_density],
        v_axis[::v_density, ::v_density],
        alpha=0.8
    )
    cmap = colors.ListedColormap(['black', 'black'])
    bounds = [0, 0, 0]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.contour(
        x_axis,
        y_axis,
        psi(x_axis, y_axis),
        levels=l_density,
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    plt.axis('equal')

    centre = plt.Circle((0, 0), 0.95, color='gray')
    plt.gca().add_patch(centre)

# speed over time (dv/dt=Sf*v)
plt.rcParams["figure.autolayout"] = True

def speed(x):
    return np.e**(sum_f * x)

x = np.linspace(-10, 10, 100)
plt.plot(
    x, speed(x),
    color='black',
    alpha=0.8
)
plt.ylabel("v [m/s]")
plt.xlabel("t [s]")
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
plt.grid()

def v_lim(x):
    return -5000

t1 = np.arange(-10, 10, 0.001)
plt.plot(
    t1,
    np.full(t1.shape, v_lim(t1)),
    alpha=0.6, color='gray')
plt.text(-9.8, 15000, 'v lim', color='black')

# flow lines functions
ball = lambda y, x : y - y / (x**2 + y**2)
ball_mask = lambda y, x : x**2 + y**2 < 0.95
flow(ball, ball_mask)

plt.show()
