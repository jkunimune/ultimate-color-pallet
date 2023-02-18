from math import hypot, atan2, pi, sqrt, degrees, radians
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

Triplet = tuple[float, float, float]


def main():
	# define some colormaps
	jet = matplotlib.cm.get_cmap("jet")

	blue_purple_pink = helical_colormap(35, 65, -100, 20, 33, (0, 0))
	black_green_yellow = helical_colormap_from_endpoints((1, 0, 0), (90, -3, 32), -1)
	black_orange_yellow = helical_colormap_from_endpoints((1, 0, 0), (90, -3, 32), +1)
	blue_gray_red = helical_colormap_from_endpoints((50, 0, 0), (75, 25, 5), -20, mirror=True)
	# blue_gray_orange = helical_colormap_from_endpoints((50, 0, 0), (85, 3, 29), 500, mirror=True)
	white_violet = linear_colormap((98, 0, 0), (25, 15, -25))
	white_aqua = helical_colormap_from_endpoints((98, 0, 0), (25, -11, -15), 1)
	blue_white_red = diverging_colormap((98, 0, 0), -60, 29, 45, -105)
	roygbivp = cyclic_colormap(22, 90, 25, -45)

	# 1. monotonic lightness
	fig, axes = plt.subplots(2, 1, figsize=(4, 2))
	plot_J(axes[0], black_green_yellow)
	plot_J(axes[1], jet)
	save(fig, "figure1.png")

	# 2. perceptually uniform
	fig, axes = plt.subplots(2, 1, figsize=(4, 2))
	plot_comb(axes[0], white_violet)
	plot_comb(axes[1], jet)
	save(fig, "figure2.png")

	# 3. simple hue progression
	fig, axes = plt.subplots(1, 2, figsize=(4, 2), subplot_kw={"projection": "polar"})
	set_grids(axes, theta=np.linspace(0, 360, 8, endpoint=False), r=[])
	plot_Ch(axes[0], blue_purple_pink)
	plot_Ch(axes[1], jet)
	save(fig, "figure3.png")

	# 4. neutral zero
	fig, axes = plt.subplots(1, 1, figsize=(sqrt(8), sqrt(8)), subplot_kw={"projection": "polar"})
	set_grids([axes], theta=[], r=40*np.array([2/11, 5/11, 8/11]))
	plot_Ch(axes, black_green_yellow)
	plot_Ch(axes, blue_gray_red)
	plot_Ch(axes, white_violet, r_max=40)
	save(fig, "figure4.png")

	# 5. consistent usage
	fig, axes = plt.subplots(1, 3, figsize=(4.6, 1.7))
	clear_ticks(axes)
	fig.tight_layout()
	x_edges = y_edges = np.linspace(-1, 1, 200)
	X, Y = np.meshgrid((x_edges[1:] + x_edges[:-1])/2, (y_edges[1:] + y_edges[:-1])/2)

	# 5a. bivariate gaussian
	P = np.exp(-(X**2 + 0.8*X*Y + Y**2)/.2)
	axes[0].imshow(P, origin="lower", cmap=white_aqua)

	# 5b. capacitor potential field
	V = np.zeros(X.shape)
	L = .20
	a = .50
	for n in range(1000):
		b_pm = [2*n*L + abs(Y + sign*L) for sign in [1, -1]]
		z_p, z_m = [1/2*((b**2 + (X + a)**2)**(1/2) + (b**2 + (X - a)**2)**(1/2)) for b in b_pm]
		V += np.arcsin(a/z_m) - np.arcsin(a/z_p)
	axes[1].imshow(V, origin="lower", cmap=blue_white_red)

	# 5c. occluded emission
	E = np.exp(-((X + .00)**2 + (Y - .02)**2)/.27)
	ρL = np.exp(-((X + .10)**2 + (Y + .02)**2)/.18)
	T = np.exp(-ρL/0.4)
	axes[2].imshow(E*T, origin="lower", cmap=black_orange_yellow)

	save(fig, "figure5.png")

	# 6a. diverging colormap
	fig, axes = plt.subplots(1, 1, figsize=(4, 1))
	plot_J(axes, blue_white_red)
	save(fig, "figure6.png")

	# 6b. cyclic colormap
	fig, axes = plt.subplots(1, 1, figsize=(4, 1))
	plot_J(axes, roygbivp)
	save(fig, "figure7.png")

	plt.show()


def plot_J(axes: Axes, cmap: Colormap):
	clear_ticks([axes])
	set_grids([axes], x=[], y=np.linspace(0, 100, 5)[1:-1])
	v = np.linspace(0, 1, 401)
	rgb = cmap(v)[:, :3]
	J, _, _ = cspace_convert(rgb, "sRGB1", "CAM02-UCS").T
	axes.scatter(v, J, c=v, s=20, cmap=cmap, zorder=2)
	axes.set_xlim(0, 1)
	axes.set_ylim(0, 100)


def plot_Ch(axes: Axes, cmap: Colormap, r_max=50):
	clear_ticks([axes])
	v = np.linspace(0, 1, 401)
	rgb = cmap(v)[:, :3]
	_, a, b = cspace_convert(rgb, "sRGB1", "CAM02-UCS").T
	C, h = np.hypot(a, b), np.arctan2(b, a)
	axes.scatter(h, C, c=v, s=20, cmap=cmap, zorder=2)
	axes.set_rlim(0, r_max)


def plot_comb(axes: Axes, cmap: Colormap):
	clear_ticks([axes])
	x_edges = np.linspace(0, 1, 201)
	y_edges = np.linspace(0, 1, 212)
	X, Y = np.meshgrid((x_edges[1:] + x_edges[:-1])/2, (y_edges[1:] + y_edges[:-1])/2)
	V = X + Y**2*np.sin(X*2*pi*40)/20
	axes.imshow(V, origin="lower", aspect="auto", cmap=cmap)


def clear_ticks(axeses: list[Axes]):
	for axes in axeses:
		axes.xaxis.set_ticks_position("none")
		axes.yaxis.set_ticks_position("none")
		axes.get_xaxis().set_ticklabels([])
		axes.get_yaxis().set_ticklabels([])


def set_grids(axeses: list[Axes],
              x: Optional[Sequence[float]] = None, y: Optional[Sequence[float]] = None,
              theta: Optional[Sequence[float]] = None, r: Optional[Sequence[float]] = None):
	for axes in axeses:
		if x is not None:
			axes.set_xticks(x)
			axes.grid(visible=True, axis="x")
		if y is not None:
			axes.set_yticks(y)
			axes.grid(visible=True, axis="y")
		if theta is not None:
			axes.set_thetagrids(theta)
		if r is not None:
			axes.set_rgrids(r)


def linear_colormap(start: Triplet, end: Triplet) -> Colormap:
	v = np.linspace(0, 1, 256)
	Jab = start + v[:, None]*np.subtract(end, start)
	rgb = cspace_convert(Jab, "CAM02-UCS", "sRGB1")
	rgb = coerce_in_gamut(rgb)
	return ListedColormap(np.hstack([rgb, np.ones((v.size, 1))]))

def diverging_colormap(start: Triplet, ΔJ: float, r: float, θ_plus: float, θ_minus: float) -> Colormap:
	v = np.linspace(0, 1, 256)
	Jab_plus = start + v[:, None]*np.array([ΔJ, r*np.cos(radians(θ_plus)), r*np.sin(radians(θ_plus))])
	Jab_minus = start + v[:, None]*np.array([ΔJ, r*np.cos(radians(θ_minus)), r*np.sin(radians(θ_minus))])
	rgb = cspace_convert(np.concatenate([Jab_minus[::-1], Jab_plus]), "CAM02-UCS", "sRGB1")
	rgb = coerce_in_gamut(rgb)
	return ListedColormap(np.hstack([rgb, np.ones((rgb.shape[0], 1))]))

def cyclic_colormap(J_min: float, J_max: float, r: float, phase: float) -> Colormap:
	v = np.linspace(0.33*pi, 2.33*pi, 256, endpoint=False)%(2*pi)
	δ = .4
	slope = (J_max - J_min)/pi
	J_true_max = J_max - δ*slope/2
	J_true_min = J_min + δ*slope/2
	J = np.where(
		v < δ, J_true_min + slope*δ/2*(v/δ)**2,
		np.where(
			v < pi - δ, J_min + slope*v,
			np.where(
				v < pi + δ, J_true_max - slope*δ/2*((v - pi)/δ)**2,
				np.where(
					v < 2*pi - δ, J_min + slope*(2*pi - v),
					J_true_min + slope*δ/2*((2*pi - v)/δ)**2))))
	a = r*np.cos(v + radians(phase))
	b = r*np.sin(v + radians(phase))
	rgb = cspace_convert(np.transpose([J, a, b]), "CAM02-UCS", "sRGB1")
	rgb = coerce_in_gamut(rgb)
	return ListedColormap(np.hstack([rgb, np.ones((v.size, 1))]))

def helical_colormap(J_start: float, J_end: float, θ_start: float, θ_end: float,
                     r: float, ab_center: tuple[float, float]) -> Colormap:
	v = np.linspace(0, 1, 256)
	J = J_start + v*(J_end - J_start)
	θ = np.radians(θ_start + v*(θ_end - θ_start))
	a = ab_center[0] + r*np.cos(θ)
	b = ab_center[1] + r*np.sin(θ)
	rgb = cspace_convert(np.transpose([J, a, b]), "CAM02-UCS", "sRGB1")
	rgb = coerce_in_gamut(rgb)
	return ListedColormap(np.hstack([rgb, np.ones((v.size, 1))]))

def helical_colormap_from_endpoints(start: Triplet, end: Triplet, center_offset: float,
                                    mirror: bool = False) -> Colormap:
	J_start, J_end = start[0], end[0]
	ab_midpoint = np.mean([start, end], axis=0)[1:]
	ab_direction = np.subtract(end, start)[1:]
	axis = np.array([-ab_direction[1], ab_direction[0]])
	ab_center = ab_midpoint + center_offset*axis/np.linalg.norm(axis)
	θ_start = degrees(atan2(start[2] - ab_center[1], start[1] - ab_center[0]))
	θ_end = degrees(atan2(end[2] - ab_center[1], end[1] - ab_center[0]))
	if center_offset > 0:
		while θ_end < θ_start:
			θ_end += 360
	else:
		while θ_end > θ_start:
			θ_end -= 360
	if mirror:
		θ_start = 2*θ_start - θ_end
		J_start = 2*J_start - J_end
	r = hypot(end[1] - ab_center[0], end[2] - ab_center[1])
	return helical_colormap(J_start, J_end, θ_start, θ_end, r, ab_center)


def coerce_in_gamut(rgb: NDArray[float]) -> NDArray[float]:
	invalid = (rgb < 0) | (rgb > 1) | np.isnan(rgb)
	if np.any(invalid):
		print(f"warning: the requested colormap would exit the sRGB gamut at "
		      f"v={np.nonzero(invalid)[0]/rgb.shape[0]}.")
		rgb = np.minimum(1, np.maximum(0, rgb))
		rgb[np.isnan(rgb)] = 0
	return rgb


def save(fig: Figure, filename: str):
	fig.tight_layout()
	fig.savefig(filename, transparent=True, dpi=300)


if __name__ == "__main__":
	main()
