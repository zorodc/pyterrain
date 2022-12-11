#!/usr/bin/env python3.7

# perlin noise (2d)

import numpy as np
from math        import sin, cos, floor, pi
from random      import seed, random

#serp = lambda a0, a1, w: (a1-a0) * (3.0 - w*2.0) * w * w + a0
lerp = lambda a0, a1, w: a0 + (a1-a0) * w
serp = lerp

R = random()

def gradient(ix, iy):
    #r = hash((ix, iy, R))
    seed((ix, iy, R));  r = random()*2*pi
    x = cos(r)
    y = sin(r)

    return np.array([x, y])

def dotgradient(ix, iy, x, y):
    gv = gradient(ix, iy)
    dv = np.array([x-ix, y-iy])
    mg = (np.linalg.norm(gv)*np.linalg.norm(dv))
    if mg < 0.001: mg = 0.001
    return np.dot(gv, dv)/mg

def perlin(x, y):
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    #fade = lambda t: t

    x0 = floor(x)
    x1 = x0+1
    y0 = floor(y)
    y1 = y0+1

    wx = fade(x - float(x0))        # find interp weights
    wy = fade(y - float(y0))

    c00 = dotgradient(x0, y0, x, y) # obtain values
    c01 = dotgradient(x0, y1, x, y)
    c10 = dotgradient(x1, y0, x, y)
    c11 = dotgradient(x1, y1, x, y)
    c_0 = serp(c00, c10, wx)        # eliminating x dim
    c_1 = serp(c01, c11, wx)
    c__ = serp(c_0, c_1, wy)        # eliminating y dim

    return (c__+1)/2.

def octave(x, y, o, p):
    t = 0.
    f = 1 # frequency in use
    a = 1 # amplitude in use
    s = 0 # amplitude sum

    for i in range(o):
        t += perlin(x*f, y*f)*a
        s += a
        a *= p
        f *= 2

    return t/s

def island(m, p, r=.5, s=1.):
    # p, a point in a unit circle centered around 0.
    d = (p[0]**2 + p[1]**2)**.5
    d = d - r # TODO: target of d=1 should probably be zero
    d = d if d < 1. else 1.
    d = d if d > 0. else 0.
    if ((p[0]**2 + p[1]**2)**.5) >= r:
        m = (1-d)*s*m + (1-s)*m

    return m

def assign(m, palette):
    k = min(k for k in palette.keys() if k > m)
    return list(map(lambda x : x / 255, palette[k]))

import matplotlib.pyplot as plt

island_palette = {
    .20  : ( 17,173,193),
    .28  : (247,182,158),
    .38  : ( 91,179, 97),
    .48  : ( 30,136,117),
    .55  : ( 96,108,129),
    9.   : (255,255,255)
}

dim = 50
img = np.zeros((dim, dim, 3))
miv = 1; mav = 0; avv = 0 # currently for debugging affects of octaves on value distribution
mav2=0
c = (dim//2, dim//2)
for xd in range(dim):
    for yd in range(dim):
        #m = perlin(float(xd)/dim*13, float(yd)/dim*13)
        m = octave(float(xd)/dim*3, float(yd)/dim*3, 6, .5)
        miv = min(miv, m); mav = max(mav, m); avv += m

        m = island(m, ((xd/dim-.5)*3, (yd/dim-.5)*3))
        mav2 = max(mav2, m)

        img[xd][yd] = np.array(assign(m, island_palette))

avv /= dim*dim
print('min:', miv)
print('max:', mav, mav2)
print('avg:', avv)

plt.imshow(img)
plt.show()
