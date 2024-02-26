import math
import time
import timeit

import numpy as np
from numpy.typing import NDArray
import pygame
import pygame.gfxdraw
import numba
from numba import int64, float64
import pygame.image


# def interpolate(i0, d0, i1, d1):
#     if i0 == i1:
#         yield d0
#         return
#
#     m = (d1 - d0) / (i1 - i0)
#     d = d0
#
#     for i in range(i0, i1):
#         yield int(d)
#         d += m
#
#     return


@numba.njit(cache=True)
def pack_rgb(r, g, b):
    return r << 16 | g << 8 | b


@numba.njit(cache=True)
def interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return [d0]

    m = (d1 - d0) / (i1 - i0)
    d = d0

    values = []

    for i in range(i0, i1 + 1):
        values.append(int(d))
        d += m

    return values


# @numba.njit(float64[:](int64, float64, int64, float64))
@numba.njit(cache=True)
def interpolate_float(i0, d0, i1, d1):
    if i0 == i1:
        return np.array([d0], dtype=np.float64)

    m = (d1 - d0) / float(i1 - i0)
    d = d0

    length = (i1 + 1) - i0

    values = np.empty((length,), dtype=np.float64)

    for i in range(length):
        values[i] = d
        d += m

    return values


@numba.njit(cache=True)
def edge_interpolate(y0, v0, y1, v1, y2, v2):
    v01 = interpolate(y0, v0, y1, v1)
    v12 = interpolate(y1, v1, y2, v2)
    v02 = interpolate(y0, v0, y2, v2)
    del v01[-1]
    return v02, v01 + v12


@numba.njit(cache=True)
def edge_interpolate_float(y0, v0, y1, v1, y2, v2):
    v01 = interpolate_float(y0, v0, y1, v1)
    v12 = interpolate_float(y1, v1, y2, v2)
    v02 = interpolate_float(y0, v0, y2, v2)
    if v01.shape[0] == 1:
        return v02, v12
    return v02, v01[:-1] + v12


@numba.njit(cache=True)
def draw_triangle(pixels: NDArray, p0, p1, p2, colour: int):
    # adapted from https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    # and https://gabrielgambetta.com/computer-graphics-from-scratch/demos/raster-12.html
    if p1[1] < p0[1]:
        p0, p1 = p1, p0
        # uv0, uv1 = uv1, uv0
    if p2[1] < p0[1]:
        p0, p2 = p2, p0
        # uv0, uv2 = uv2, uv0
    if p2[1] < p1[1]:
        p1, p2 = p2, p1
        # uv1, uv2 = uv2, uv1

    # x01 = interpolate(p0[1], p0[0], p1[1], p1[0])
    # x12 = interpolate(p1[1], p1[0], p2[1], p2[0])
    # x02 = interpolate(p0[1], p0[0], p2[1], p2[0])
    #
    # # x01 = x01[:-1]
    # del x01[-1]
    # x012 = x01 + x12

    x02, x012 = edge_interpolate(p0[1], p0[0], p1[1], p1[0], p2[1], p2[0])
    # uz02, uz012 =

    # print(x012)

    m = math.floor(len(x02) / 2)
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
    else:
        x_left = x012
        x_right = x02

    # print(p0, p1, p2)
    # print(len(x02), len(x012))
    # print(len(x_left), len(x_right))

    for y in range(p0[1], p2[1] + 1):
        # pixels[x_left[y - p0[1]]:x_right[y - p0[1]], y] = colour
        # print(y - p0[1], y - p0[1])
        for x in range(x_left[y - p0[1]], x_right[y - p0[1]]):
            pixels[x, y] = colour


@numba.njit(cache=True)
def draw_triangle_textured(pixels: NDArray, p0, p1, p2, uv0, uv1, uv2, texture):
    # adapted from https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    # and https://gabrielgambetta.com/computer-graphics-from-scratch/demos/raster-12.html
    if p1[1] < p0[1]:
        p0, p1 = p1, p0
        uv0, uv1 = uv1, uv0
    if p2[1] < p0[1]:
        p0, p2 = p2, p0
        uv0, uv2 = uv2, uv0
    if p2[1] < p1[1]:
        p1, p2 = p2, p1
        uv1, uv2 = uv2, uv1

    x02, x012 = edge_interpolate(p0[1], p0[0], p1[1], p1[0], p2[1], p2[0])
    u02, u012 = edge_interpolate_float(p0[1], uv0[0], p1[1], uv1[0], p2[1], uv2[0])
    v02, v012 = edge_interpolate_float(p0[1], uv0[1], p1[1], uv1[1], p2[1], uv2[1])

    m = math.floor(len(x02) / 2)
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
        u_left = u02
        u_right = u012
        v_left = v02
        v_right = v012
    else:
        x_left = x012
        x_right = x02
        u_left = u012
        u_right = u02
        v_left = v012
        v_right = v02

    # print(p0, p1, p2)
    # print(len(x02), len(x012))
    # print(len(x_left), len(x_right))

    texture_x, texture_y = texture.shape

    for y in range(p0[1], p2[1] + 1):
        # pixels[x_left[y - p0[1]]:x_right[y - p0[1]], y] = colour
        # print(y - p0[1], y - p0[1])
        y_actual = y - p0[1]
        x_l = x_left[y_actual]
        x_r = x_right[y_actual]

        u_segment = interpolate_float(x_l, u_left[y_actual], x_r, u_right[y_actual])
        v_segment = interpolate_float(x_l, v_left[y_actual], x_r, v_right[y_actual])
        for x in range(x_l, x_r):
            x_actual = x - x_l
            # pixels[x, y] = 255
            sample_x = int(u_segment[x_actual] * texture_x)
            sample_y = int(v_segment[x_actual] * texture_y)
            sample = texture_pixels[sample_x, sample_y]
            pixels[x, y] = sample
            # pixels[x, y] = pack_rgb(int(u_segment[x_actual] * 255), int(v_segment[x_actual] * 255), 0)


@numba.njit(cache=True)
def draw_triangle_interpolated_depth(pixels, z_buffer, p0, p1, p2, z0, z1, z2):
    # adapted from https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    # and https://gabrielgambetta.com/computer-graphics-from-scratch/demos/raster-12.html
    if p1[1] < p0[1]:
        p0, p1 = p1, p0
        z0, z1 = z1, z0
    if p2[1] < p0[1]:
        p0, p2 = p2, p0
        z0, z2 = z2, z0
    if p2[1] < p1[1]:
        p1, p2 = p2, p1
        z1, z2 = z2, z1

    x02, x012 = edge_interpolate(p0[1], p0[0], p1[1], p1[0], p2[1], p2[0])
    z02, z012 = edge_interpolate_float(p0[1], z0, p1[1], z1, p2[1], z2)

    m = math.floor(len(x02) / 2)
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
        z_left = z02
        z_right = z012
    else:
        x_left = x012
        x_right = x02
        z_left = z012
        z_right = z02

    # print(p0, p1, p2)
    # print(len(x02), len(x012))
    # print(len(x_left), len(x_right))

    for y in range(p0[1], p2[1] + 1):
        # pixels[x_left[y - p0[1]]:x_right[y - p0[1]], y] = colour
        # print(y - p0[1], y - p0[1])
        y_actual = y - p0[1]
        x_l = x_left[y_actual]
        x_r = x_right[y_actual]

        z_segment = interpolate_float(x_l, z_left[y_actual], x_r, z_right[y_actual])
        # v_segment = interpolate_float(x_l, v_left[y_actual], x_r, v_right[y_actual])
        for x in range(x_l, x_r):
            x_actual = x - x_l
            z = z_segment[x_actual]
            if z < z_buffer[x, y]:
                z_buffer[x, y] = z
                v = int(z * 255.0)
                pixels[x, y] = pack_rgb(v, v, v)
                # pixels[x, y] = 255


@numba.njit(cache=True)
def draw_triangle_interpolated(pixels, z_buffer, colour, p0, p1, p2, z0, z1, z2):
    # adapted from https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    # and https://gabrielgambetta.com/computer-graphics-from-scratch/demos/raster-12.html
    if p1[1] < p0[1]:
        p0, p1 = p1, p0
        z0, z1 = z1, z0
    if p2[1] < p0[1]:
        p0, p2 = p2, p0
        z0, z2 = z2, z0
    if p2[1] < p1[1]:
        p1, p2 = p2, p1
        z1, z2 = z2, z1

    x02, x012 = edge_interpolate(p0[1], p0[0], p1[1], p1[0], p2[1], p2[0])
    z02, z012 = edge_interpolate_float(p0[1], z0, p1[1], z1, p2[1], z2)

    m = math.floor(len(x02) / 2)
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
        z_left = z02
        z_right = z012
    else:
        x_left = x012
        x_right = x02
        z_left = z012
        z_right = z02

    # print(p0, p1, p2)
    # print(len(x02), len(x012))
    # print(len(x_left), len(x_right))

    for y in range(p0[1], p2[1] + 1):
        # pixels[x_left[y - p0[1]]:x_right[y - p0[1]], y] = colour
        # print(y - p0[1], y - p0[1])
        y_actual = y - p0[1]
        x_l = x_left[y_actual]
        x_r = x_right[y_actual]

        z_segment = interpolate_float(x_l, z_left[y_actual], x_r, z_right[y_actual])
        # v_segment = interpolate_float(x_l, v_left[y_actual], x_r, v_right[y_actual])
        for x in range(x_l, x_r):
            x_actual = x - x_l
            z = z_segment[x_actual]
            if z < z_buffer[x, y]:
                z_buffer[x, y] = z
                pixels[x, y] = colour
                # pixels[x, y] = 255


@numba.njit(cache=True)
def draw_triangle_textured_buffered(pixels, z_buffer, texture, p0, p1, p2, z0, z1, z2, uv0, uv1, uv2):
    # adapted from https://gabrielgambetta.com/computer-graphics-from-scratch/07-filled-triangles.html
    # and https://gabrielgambetta.com/computer-graphics-from-scratch/demos/raster-12.html
    if p1[1] < p0[1]:
        p0, p1 = p1, p0
        uv0, uv1 = uv1, uv0
        z0, z1 = z1, z0
    if p2[1] < p0[1]:
        p0, p2 = p2, p0
        uv0, uv2 = uv2, uv0
        z0, z2 = z2, z0
    if p2[1] < p1[1]:
        p1, p2 = p2, p1
        uv1, uv2 = uv2, uv1
        z1, z2 = z2, z1

    x02, x012 = edge_interpolate(p0[1], p0[0], p1[1], p1[0], p2[1], p2[0])
    z02, z012 = edge_interpolate_float(p0[1], z0, p1[1], z1, p2[1], z2)
    u02, u012 = edge_interpolate_float(p0[1], uv0[0], p1[1], uv1[0], p2[1], uv2[0])
    v02, v012 = edge_interpolate_float(p0[1], uv0[1], p1[1], uv1[1], p2[1], uv2[1])

    m = math.floor(len(x02) / 2)
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
        z_left = z02
        z_right = z012
        u_left = u02
        u_right = u012
        v_left = v02
        v_right = v012
    else:
        x_left = x012
        x_right = x02
        z_left = z012
        z_right = z02
        u_left = u012
        u_right = u02
        v_left = v012
        v_right = v02

    # print(p0, p1, p2)
    # print(len(x02), len(x012))
    # print(len(x_left), len(x_right))

    texture_x, texture_y = texture.shape

    for y in range(p0[1], p2[1] + 1):
        # pixels[x_left[y - p0[1]]:x_right[y - p0[1]], y] = colour
        # print(y - p0[1], y - p0[1])
        y_actual = y - p0[1]
        x_l = x_left[y_actual]
        x_r = x_right[y_actual]

        u_segment = interpolate_float(x_l, u_left[y_actual], x_r, u_right[y_actual])
        v_segment = interpolate_float(x_l, v_left[y_actual], x_r, v_right[y_actual])
        z_segment = interpolate_float(x_l, z_left[y_actual], x_r, z_right[y_actual])
        # v_segment = interpolate_float(x_l, v_left[y_actual], x_r, v_right[y_actual])
        for x in range(x_l, x_r):
            x_actual = x - x_l
            z = z_segment[x_actual]
            if z < z_buffer[x, y]:
                z_buffer[x, y] = z
                # v = int(z * 255.0)
                sample_x = int(u_segment[x_actual] * texture_x)
                sample_y = int(v_segment[x_actual] * texture_y)
                sample = texture_pixels[sample_x, sample_y]
                pixels[x, y] = sample
                # pixels[x, y] = pack_rgb(v, v, v)
                # pixels[x, y] = 255


def clear(pixels):
    pixels[:] = 0


if __name__ == '__main__':
    pygame.init()

    screen = pygame.display.set_mode((500, 500))
    z_buffer = np.ones(screen.get_size(), dtype=np.float64)

    texture = pygame.image.load("texture.jpg").convert()
    texture_pixels_view = pygame.surfarray.pixels2d(texture)
    texture_pixels = np.copy(texture_pixels_view)
    del texture_pixels_view
    # print(texture_pixels.shape)

    start = time.time()
    runtime = 30
    frame_count = 0

    p0 = np.array([10, 10], dtype=np.int64)
    p1 = np.array([490, 10], dtype=np.int64)
    p2 = np.array([490, 490], dtype=np.int64)
    # p0 = np.array([10, 10], dtype=np.int64)
    # p1 = np.array([50, 10], dtype=np.int64)
    # p2 = np.array([50, 50], dtype=np.int64)
    uv0 = np.array([0, 0], dtype=np.float32)
    uv1 = np.array([1, 0], dtype=np.float32)
    uv2 = np.array([1, 1], dtype=np.float32)

    while time.time() < start + runtime:
        pygame.event.pump()

        # screen.fill((255, 255, 255))

        # pixels[:] = 0b111111110000000000000000

        pixels = pygame.surfarray.pixels2d(screen)

        clear(pixels)

        # draw_triangle(pixels, p0, p1, p2, 0b111111110000000000000000)
        # draw_triangle_textured(pixels, p0, p1, p2, uv0, uv1, uv2, texture_pixels)
        z_buffer[:] = 1
        # draw_triangle_textured_buffered(pixels, z_buffer, texture_pixels,
        #                            p0, p1, p2,
        #                            1.0, 0.5, 0.0,
        #                            uv0, uv1, uv2)
        draw_triangle_interpolated(pixels, z_buffer, pack_rgb(255, 0, 0), p0, p1, p2, 0, 0.5, 1)
        draw_triangle_interpolated(pixels, z_buffer, pack_rgb(0, 255, 0), p0, p1, p2, 1, 0.5, 0)
        # draw_triangle_interpolated(pixels, z_buffer, p0, p1, p2, 0, 0.5, 1)
        # pygame.gfxdraw.filled_trigon(screen, p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], (255, 0, 0))

        del pixels

        pygame.display.flip()

        frame_count += 1

    print(f"mean fps: {frame_count / runtime}")


# draw_triangle 50s mean fps: 1825.02
# gfxdraw 50s mean fps: 2040.74
# draw_triangle_textured mean fps: 291.56 (480 x 480) triangle, ~800x800 texture
# draw_triangle_interpolated mean fps: 199.1 5 draw calls with no fragment draws (z_buffer[:] = 0)
# draw_triangle_interpolated mean fps: 148.7 5 draw calls (z-buffer cleared)
