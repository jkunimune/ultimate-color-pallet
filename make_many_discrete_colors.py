from math import pi, sqrt, cos, sin, hypot, atan2, degrees, radians, floor

from colorspacious import cspace_convert
from matplotlib import pyplot as plt

NUM_COLORS = 36
RADIUS = 30
WHITEPOINT = (-8, -8)
BLANK_LIGHTNESS = 95
MAX_LIGHTNESS = 73
MIN_LIGHTNESS = 60

GOLDEN_RATIO = (1 + sqrt(5))/2

Jrθ_colors = [(BLANK_LIGHTNESS, 0, 0)]
for i in range(NUM_COLORS):
	lightness = MIN_LIGHTNESS + (MAX_LIGHTNESS - MIN_LIGHTNESS)*(i/(1 + GOLDEN_RATIO) % 1)
	Jrθ_colors.append((lightness, RADIUS, 360/NUM_COLORS*i))

rgb_colors = []
for J, r, θ in Jrθ_colors:
	print(f"J = {J:.1f}, r = {r:.1f}, θ = {θ:.1f}°")
	a = WHITEPOINT[0] + r*cos(radians(θ))
	b = WHITEPOINT[1] + r*sin(radians(θ))
	C, h = hypot(a, b), degrees(atan2(b, a))
	r, g, b = cspace_convert((J, C, h), "JCh", "sRGB1")
	assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1
	rgb_colors.append((r, g, b))

for r, g, b in rgb_colors:
	print(f'"#{floor(256*r):02x}{floor(256*g):02x}{floor(256*b):02x}",', end=" ")

for (J, _, θ), (r, g, b) in zip(Jrθ_colors, rgb_colors):
	plt.scatter(x=θ, y=J, c=[(r, g, b)])
plt.show()
