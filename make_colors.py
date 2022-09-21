from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
import numpy as np

LAB = np.array([
	# SANI RANG
	# [-0.464101, -0.267949, 2],
	# [ 0.464101, -0.267949, 1],
	# [ 0.000000,  0.535898, 0],
	# LIMI RANG
	# [-0.7,  0.0, 2],
	# [ 0.7,  0.0, 7],
	# [ 0.0, -0.7, 4],
	# [ 0.0,  0.7, 1],
	# [ 0.0,  0.0, 5],
	# SETI RANG
	[ 0.700000,  0.000000, 4],
	[ 0.216311,  0.665739, 2],
	[-0.566311,  0.411449, 1],
	[-0.566311, -0.411449, 3],
	[ 0.216311, -0.665739, 5],
	[ 0.000000,  0.000000, 0],
	[ 0.000000,  0.000000, 6],
	# UNNOLI RANG
	# [-0.415055,  0.609910, 2],
	# [ 0.262258, -0.689552, 6],
	# [ 0.415055,  0.609910, 5],
	# [-0.262258, -0.689552, 3],
	# [-0.715460,  0.179938, 4],
	# [ 0.654207, -0.340990, 9],
	# [ 0.715460,  0.179938, 1],
	# [-0.654207, -0.340990, 8],
	# [ 0.000000, -0.235306, 0],
	# [ 0.000000,  0.289211, 7],
])

# SANI RANGI
# θ = 3.0
# LIMI RANG
# θ = 2.0
# SETI RANG
# θ = 2.9
# UNNOLI RANG
θ = 2.2
R = 60
MIN, MAX = 40, 80

RGB = []
for x, y, z in LAB:
	x, y = x*np.cos(θ) - y*np.sin(θ), x*np.sin(θ) + y*np.cos(θ)
	lab = LabColor(np.interp(z, [0, len(LAB)-1], [MIN, MAX]), (x+.1)*R, y*R)
	print(lab)
	RGB.append(convert_color(lab, sRGBColor).get_upscaled_value_tuple())
	# print(f"{int(z):d}: ({RGB[-1][0]/255:.3f}, {RGB[-1][1]/255:.3f}, {RGB[-1][2]/255:.3f})")
	# print(f"{int(z):d}: rgb({RGB[-1][0]:d}, {RGB[-1][1]:d}, {RGB[-1][2]:d})")
	print(f"{int(z):d}: #{RGB[-1][0]:02X}{RGB[-1][1]:02X}{RGB[-1][2]:02X}")

plt.figure(figsize=(6, 6))
plt.scatter(LAB[:,0], LAB[:,1], s=6000, c=np.array(RGB)/255)
for (x, y, z), (r, g, b) in zip(LAB, RGB):
	plt.text(x, y, f"{int(z):d}: #{r:02X}{g:02X}{b:02X}", horizontalalignment='center')
plt.axis('square')
plt.axis([-1, 1, -1, 1])
plt.show()
