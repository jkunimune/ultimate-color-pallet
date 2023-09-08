# make_color_maps.py

# generate a set of several perceptually uniform black-to-color maps that go well together

from colorspacious import cspace_convert
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos


STARTING_HUE = 10
NUM_HUES = 9
MAX_CHROMA = 40
MAX_LIGHTNESS = 80

def define_polar_color(hue, chroma, lightness) -> tuple[int, int, int]:
	red, green, blue = cspace_convert((lightness, chroma, hue), "JCh", "sRGB1")
	if red >= 1 or red < 0 or green >= 1 or green < 0 or blue >= 1 or blue < 0:
		raise ValueError(f"{chroma:.2f} is too saturated for angle {hue:.2f} at "
		                 f"lightness {lightness:.2f} ({red:.4f}, {green:.4f}, {blue:.4f})")
	return (red, green, blue)


maps = np.empty((NUM_HUES, 101, 3), dtype=float)
for i, hue in enumerate(np.linspace(0, 360, NUM_HUES, endpoint=False) + STARTING_HUE):
	max_lightness = MAX_LIGHTNESS
	while True:
		try:
			define_polar_color(hue, MAX_CHROMA, max_lightness)
		except ValueError:
			if max_lightness < 10:
				raise
			max_lightness -= 2
		else:
			break
	print(f"{max_lightness:.1f}")

	for j, lightness in enumerate(np.linspace(0, max_lightness, 101)):
		chroma = lightness/max_lightness*MAX_CHROMA
		red, green, blue = define_polar_color(hue, chroma, lightness)
		maps[i, j, :] = [red, green, blue]

		plt.scatter(lightness, hue, c=[(red, green, blue)])

	np.savetxt(f"cmap_{hue:.0f}.csv", maps[i, :, :], "%.3f")

plt.show()
