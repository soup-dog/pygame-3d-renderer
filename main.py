from __future__ import annotations

from dataclasses import dataclass
import math
import time
from enum import Enum
from typing import List, Tuple
import pickle
import hashlib
import timeit
import weakref

import pygame
import pygame.gfxdraw
import numpy as np
from numpy.typing import NDArray
from line_profiler_pycharm import profile

np.set_printoptions(suppress=True)
pygame.init()


def perspective_matrix(scale: float, far: float, near: float, aspect: float) -> NDArray:
    return np.array([
        [scale / aspect, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ])


# def rotation_matrix_x(angle: float) -> NDArray:
#     return np.array([
#         [1, 0, 0, 0],
#         [0, np.cos(angle), -np.sin(angle), 0],
#         [0, np.sin(angle), np.cos(angle), 0],
#         [0, 0, 0, 1],
#     ])
#
#
# def rotation_matrix_y(angle: float) -> NDArray:
#     return np.array([
#         [np.cos(angle), 0, np.sin(angle), 0],
#         [0, 1, 0, 0],
#         [-np.sin(angle), 0, np.cos(angle), 0],
#         [0, 0, 0, 1],
#     ])
#
#
# def rotation_matrix_z(angle: float) -> NDArray:
#     return np.array([
#         [np.cos(angle), -np.sin(angle), 0, 0],
#         [np.sin(angle), np.cos(angle), 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ])


def rotation_matrix_x(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)],
    ])


def rotation_matrix_y(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])


def rotation_matrix_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])


def compose_matrix(translation: NDArray, scale: NDArray, rotation: NDArray) -> NDArray:
    mat = np.identity(4)
    mat[:3, :3] = rotation
    mat[0, 3] = translation[0]
    mat[1, 3] = translation[1]
    mat[2, 3] = translation[2]
    # i do not know why this works but it does
    mat[:3, 0] *= scale[0]
    mat[:3, 1] *= scale[1]
    mat[:3, 2] *= scale[2]
    return mat


def get_scale(mat: NDArray) -> NDArray:
    return np.array([magnitude_vec3(mat[0, :3]), magnitude_vec3(mat[1, :3]), magnitude_vec3(mat[2, :3])])


def dot_vec3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def distance_squared_vec3(a: NDArray, b: NDArray) -> NDArray:
    l0 = a[0] - b[0]
    l1 = a[1] - b[1]
    l2 = a[2] - b[2]
    return l0 * l0 + l1 * l1 + l2 * l2


def magnitude_squared_vec3(v: NDArray) -> float:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


def magnitude_vec3(v: NDArray) -> float:
    return np.sqrt(magnitude_squared_vec3(v))


def cross_vec3(a: NDArray, b: NDArray) -> NDArray:
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def normalise_vec3(v: NDArray) -> NDArray:
    return v / magnitude_vec3(v)


def v3_to_column_v4(vector: NDArray) -> NDArray:
    array = np.empty((4, 1))
    array[0, 0] = vector[0]
    array[1, 0] = vector[1]
    array[2, 0] = vector[2]
    array[3, 0] = 1
    return array


def column_v4_to_v3(vector: NDArray) -> NDArray:
    return vector[:3, 0].flatten()


class Plane:
    def __init__(self, normal: NDArray, distance: float):
        self.normal: NDArray = normal
        self.distance: float = distance

    @staticmethod
    def from_point_normal(point: NDArray, normal: NDArray):
        return Plane(
            normal=normalise_vec3(normal),
            distance=(-normal[0] * point[0] - normal[1] * point[1] - normal[2] * point[2]) / magnitude_vec3(normal)
        )

    def signed_distance_to(self, point: NDArray) -> float:
        return dot_vec3(self.normal, point) + self.distance


class Frustum:
    def __init__(self, near: Plane, far: Plane, left: Plane, right: Plane, top: Plane, bottom: Plane):
        self.near: Plane = near
        self.far: Plane = far
        self.left: Plane = left
        self.right: Plane = right
        self.top: Plane = top
        self.bottom: Plane = bottom

    @staticmethod
    def from_camera(camera: Camera, fov: float, aspect: float) -> Frustum:
        # from https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling
        half_v_side = camera.far * np.tan(fov * 0.5)
        half_h_side = half_v_side * aspect
        far = camera.far * camera.front

        return Frustum(
            near=Plane.from_point_normal(camera.position + camera.near * camera.front, camera.front),  # near
            far=Plane.from_point_normal(camera.position + far, -camera.front),  # far
            left=Plane.from_point_normal(camera.position, cross_vec3(camera.up, far + camera.right * half_h_side)),
            # left
            right=Plane.from_point_normal(camera.position, cross_vec3(far - camera.right * half_h_side, camera.up)),
            # right
            top=Plane.from_point_normal(camera.position, cross_vec3(camera.right, far - camera.up * half_v_side)),
            # top
            bottom=Plane.from_point_normal(camera.position, cross_vec3(far + camera.up * half_v_side, camera.right)),
            # bottom
        )


class BoundingSphere:
    def __init__(self, centre: NDArray = None, radius: float = 0):
        if centre is None:
            centre = np.zeros((3,))
        self.centre: NDArray = centre
        self.radius: float = radius

    def above_plane(self, plane: Plane):
        return plane.signed_distance_to(self.centre) > -self.radius

    def in_frustum(self, frustum: Frustum):
        return self.above_plane(frustum.near) \
            and self.above_plane(frustum.far) \
            and self.above_plane(frustum.left) \
            and self.above_plane(frustum.right) \
            and self.above_plane(frustum.top) \
            and self.above_plane(frustum.bottom)


class AABB:
    def __init__(self, lower: NDArray = None, upper: NDArray = None):
        if lower is None:
            lower = np.zeros((3,))
        if upper is None:
            upper = np.zeros((3,))
        self.lower: NDArray = lower
        self.upper: NDArray = upper

    @staticmethod
    def from_geometry(geometry: Geometry):
        min_x = np.inf
        min_y = np.inf
        min_z = np.inf
        max_x = -np.inf
        max_y = -np.inf
        max_z = -np.inf

        for vertex in geometry.vertex_buffer.T:
            if vertex[0] < min_x:
                min_x = vertex[0]
            if vertex[1] < min_y:
                min_y = vertex[1]
            if vertex[2] < min_z:
                min_z = vertex[2]
            if vertex[0] > max_x:
                max_x = vertex[0]
            if vertex[1] > max_y:
                max_y = vertex[1]
            if vertex[2] > max_z:
                max_z = vertex[2]

        return AABB(np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z]))


class Geometry:
    def __init__(self):
        self._vertex_buffer: NDArray = np.empty((0,))
        self.index_buffer: NDArray = np.empty((0,))
        self.colour_buffer: NDArray = np.empty((0,))
        self.aabb: AABB = AABB()
        self.bounding_sphere: BoundingSphere = BoundingSphere()

    @property
    def vertex_buffer(self) -> NDArray:
        return self._vertex_buffer

    @vertex_buffer.setter
    def vertex_buffer(self, value: NDArray):
        if value.shape[0] == 3:  # if not already padded, one-time pre-pad for performance
            self._vertex_buffer = np.pad(value, [(0, 1), (0, 0)], mode="constant", constant_values=1)
        else:
            self._vertex_buffer = value
        self.aabb = AABB.from_geometry(self)
        self.bounding_sphere = self.compute_bounding_sphere()

    # def compute_aabb(self) -> Tuple[NDArray, NDArray]:
    #     min_x = np.inf
    #     min_y = np.inf
    #     min_z = np.inf
    #     max_x = -np.inf
    #     max_y = -np.inf
    #     max_z = -np.inf
    #
    #     for vertex in self._vertex_buffer.T:
    #         if vertex[0] < min_x:
    #             min_x = vertex[0]
    #         if vertex[1] < min_y:
    #             min_y = vertex[1]
    #         if vertex[2] < min_z:
    #             min_z = vertex[2]
    #         if vertex[0] > max_x:
    #             max_x = vertex[0]
    #         if vertex[1] > max_y:
    #             max_y = vertex[1]
    #         if vertex[2] > max_z:
    #             max_z = vertex[2]
    #
    #     return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])
    #     # return min_x, max_x, min_y, max_y, min_z, max_z

    def compute_bounding_sphere(self) -> BoundingSphere:
        lower = self.aabb.lower
        upper = self.aabb.upper
        # (min_x, min_y, min_z), (max_x, max_y, max_z)
        # print(min_x, min_y, min_z, max_x, max_y, max_z)

        centre = np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, (lower[2] + upper[2]) / 2])
        edge = upper - centre

        return BoundingSphere(centre, magnitude_vec3(edge))

        # r_squared = 0
        # for vertex in self.vertex_buffer:
        #     current_r_squared = distance_squared(np.array([centre_x, centre_y, centre_z]), vertex)
        #     if current_r_squared > r_squared:
        #         r_squared = current_r_squared
        #
        # return np.sqrt(r_squared)


class Object3D:
    def __init__(self):
        self._position: NDArray = np.zeros((3,), dtype=np.float64)
        self._scale: NDArray = np.ones((3,), dtype=np.float64)
        self._rotation: NDArray = np.identity(3)
        self.model_matrix = np.identity(4)
        self.model_matrix_needs_update: bool = False
        self._front: NDArray = np.array([0, 0, -1], dtype=np.float64)
        self._up: NDArray = np.array([0, 1, 0], dtype=np.float64)
        self._right: NDArray = np.array([1, 0, 0], dtype=np.float64)

    def update_model_matrix(self):
        self.model_matrix = compose_matrix(self.position, self.scale, self.rotation)
        self._front = self.rotation.dot(np.array([[0], [0], [-1]], dtype=np.float64)).flatten()
        self._up = self.rotation.dot(np.array([[0], [1], [0]], dtype=np.float64)).flatten()
        self._right = self.rotation.dot(np.array([[1], [0], [0]], dtype=np.float64)).flatten()
        self.model_matrix_needs_update = False

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        self.model_matrix_needs_update = True

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self.model_matrix_needs_update = True

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.model_matrix_needs_update = True

    @property
    def front(self):
        return self._front

    @property
    def up(self):
        return self._up

    @property
    def right(self):
        return self._right


class Mesh(Object3D):
    def __init__(self, geometry: Geometry, material):
        super().__init__()
        self.geometry: Geometry = geometry
        self.material = material
        # self._geometry_bounding_sphere_ref: weakref.ref = weakref.ref(self.geometry.bounding_sphere)
        self.bounding_sphere: BoundingSphere = geometry.bounding_sphere

    # @property
    # def bounding_sphere_needs_update(self):
    #     geometry_bounding_sphere = self._geometry_bounding_sphere_ref()
    #     return geometry_bounding_sphere is None or geometry_bounding_sphere is not self.geometry.bounding_sphere

    def update_bounding_sphere(self):
        centre = column_v4_to_v3(self.model_matrix.dot(v3_to_column_v4(self.geometry.bounding_sphere.centre)))
        max_scale = get_scale(self.model_matrix).max()
        radius = self.geometry.bounding_sphere.radius * max_scale
        # self._geometry_bounding_sphere_ref = weakref.ref(self.geometry.bounding_sphere)
        self.bounding_sphere = BoundingSphere(centre, radius)


class Camera(Object3D):
    def __init__(self, far: float, near: float, frustum_scale: float = None, fov: float = None, aspect: float = 1):
        super().__init__()
        if fov is not None:
            self.fov: float = fov
        if frustum_scale is not None:
            self.frustum_scale: float = frustum_scale
        if frustum_scale is None and fov is None:
            self._frustum_scale: float = 1
            self._fov: float = np.pi * 0.5
        self.far: float = far
        self.near: float = near
        self.aspect: float = aspect
        self.frustum: Frustum = Frustum.from_camera(self, self.fov, self.aspect)
        self.projection_matrix: NDArray = perspective_matrix(self.frustum_scale, self.far, self.near, self.aspect)
        self.view_matrix: NDArray = np.identity(4)
        self.camera_matrix: NDArray = self.projection_matrix

    @property
    def frustum_scale(self):
        return self._frustum_scale

    @frustum_scale.setter
    def frustum_scale(self, value):
        self._frustum_scale = value
        self._fov = np.arctan(1 / self._frustum_scale) * 2

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = value
        self._frustum_scale = 1 / np.tan(self._fov / 2)

    def update_projection_matrix(self):
        self.projection_matrix = perspective_matrix(self.frustum_scale, self.far, self.near, self.aspect)

    def update_view_matrix(self):
        self.update_model_matrix()
        self.view_matrix = np.linalg.inv(self.model_matrix)

    def update_camera_matrix(self):
        self.update_projection_matrix()
        self.update_view_matrix()
        self.camera_matrix = self.projection_matrix.dot(self.view_matrix)

    def update_frustum(self):
        self.frustum = Frustum.from_camera(self, self.fov, self.aspect)


class Scene:
    def __init__(self):
        self.children: List[Object3D] = []


class WindingOrder(Enum):
    CLOCKWISE = False
    ANTICLOCKWISE = True


class Renderer:
    def __init__(self):
        self.winding_order: WindingOrder = WindingOrder.ANTICLOCKWISE
        self.face_culling: bool = True

    # @staticmethod
    # def apply_transforms(vertex_buffer: NDArray, model_matrix: NDArray, view_matrix: NDArray, projection_matrix: NDArray) -> NDArray:
    #     return projection_matrix.dot(view_matrix.dot(model_matrix.dot(vertex_buffer)))

    @staticmethod
    def apply_transforms(vertex_buffer: NDArray, model_matrix: NDArray, camera_matrix: NDArray) -> NDArray:
        return camera_matrix.dot(model_matrix).dot(vertex_buffer)

    @staticmethod
    def viewport_transform(ndc: NDArray, x: float, y: float, width: float, height: float, f: float,
                           n: float) -> NDArray:
        half_width = width / 2
        half_height = height / 2

        # print(np.array([[half_width, half_height, (f - n) / 2]]).T)
        # print(np.array([[x + half_width, y + half_height, (f + n) / 2]]).T)

        return ndc * np.array([[half_width, half_height, (f - n) / 2]]).T + np.array(
            [[x + half_width, y + half_height, (f + n) / 2]]).T
        # return ndc + np.array([[2, 1, 1]]).T

    @staticmethod
    def on_surface(width, height, point: NDArray):
        return 0 < point[0] < width and 0 < point[1] < height

    def front_facing(self, a, b, c):
        n = dot_vec3(a, cross_vec3(b - a, c - a))
        return n == 0 or n > 0 != self.winding_order

    def transform(self, camera: Camera, mesh: Mesh, surface: pygame.Surface):
        clip = self.apply_transforms(
            mesh.geometry.vertex_buffer,
            mesh.model_matrix,
            camera.camera_matrix,
        )
        ndc = (clip / clip[3])[:3]
        size = surface.get_size()
        viewport = self.viewport_transform(ndc, 0, 0, size[0], size[1], camera.far, camera.near).T

        return clip, viewport

    def draw_scene(self, scene: Scene, camera: Camera, surface: pygame.surface.Surface):
        camera.update_camera_matrix()
        camera.update_frustum()

        triangles = []

        mesh_count = 0

        for obj in scene.children:
            if isinstance(obj, Mesh):
                if obj.model_matrix_needs_update:
                    obj.update_model_matrix()
                obj.update_bounding_sphere()

                # print(obj.bounding_sphere.centre, obj.bounding_sphere.radius)

                if obj.bounding_sphere.in_frustum(camera.frustum):
                    clip, viewport = self.transform(camera, obj, surface)

                    triangles.extend(
                        self.make_triangles(clip, viewport, mesh.geometry.index_buffer, mesh.geometry.colour_buffer,
                                            surface))

                    mesh_count += 1

        # print(triangles)
        triangles.sort(reverse=True, key=lambda x: max(x[0][2], x[1][2], x[2][2]))
        # triangles.sort(reverse=True, key=lambda x: x[0][2] + x[1][2] + x[2][2] + max(x[0][2], x[1][2], x[2][2]))
        # print(triangles)

        surface.lock()

        for triangle in triangles:
            s0, s1, s2, colour = triangle
            pygame.gfxdraw.filled_trigon(surface, int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]), int(s2[0]),
                                         int(s2[1]), colour)

        surface.unlock()

        return mesh_count

    def make_triangles(self, clip: NDArray, viewport: NDArray, index_buffer: NDArray, colour_buffer: NDArray,
                       surface: pygame.Surface):
        # clip, viewport = self.transform(camera, mesh, surface)

        width, height = surface.get_size()

        for i in range(0, index_buffer.shape[0], 3):
            indices = index_buffer[i:i + 3]

            c0 = clip[:, indices[0]].flatten()
            c1 = clip[:, indices[1]].flatten()
            c2 = clip[:, indices[2]].flatten()

            # if clip[2, indices[0]] > 0 or clip[2, indices[1]] > 0 or clip[2, indices[2]] > 0:
            if c0[2] > 0 or c1[2] > 0 or c2[2] > 0 and c0[3] != 0 and c1[3] != 0 and c2[3] != 0:
                s0 = viewport[indices[0]][:3]
                s1 = viewport[indices[1]][:3]
                s2 = viewport[indices[2]][:3]

                if not self.face_culling or self.front_facing(c0, c1, c2):
                    if self.on_surface(width, height, s0) and self.on_surface(width, height, s1) and self.on_surface(
                            width, height, s2):
                        yield s0, s1, s2, colour_buffer[i // 3]

    def draw_points(self, camera: Camera, mesh: Mesh, surface: pygame.Surface, radius: int = 2):
        clip, viewport = self.transform(camera, mesh, surface)

        for r in range(viewport.shape[0]):
            vertex = viewport[r]
            if clip[2, r] > 0:  # in front of camera
                screen = vertex[:2]
                pygame.gfxdraw.filled_circle(surface, int(screen[0]), int(screen[1]), radius, (255, 0, 0))

    # def draw_triangles(self, camera: Camera, mesh: Mesh, surface: pygame.Surface):
    #     clip, viewport = self.transform(camera, mesh, surface)
    #
    #     width, height = surface.get_size()
    #
    #     for i in range(0, mesh.geometry.index_buffer.shape[0], 3):
    #         indices = mesh.geometry.index_buffer[i:i + 3]
    #
    #         c0 = clip[:3, indices[0]].T
    #         c1 = clip[:3, indices[1]].T
    #         c2 = clip[:3, indices[2]].T
    #
    #         # if clip[2, indices[0]] > 0 or clip[2, indices[1]] > 0 or clip[2, indices[2]] > 0:
    #         if c0[2] > 0 or c1[2] > 0 or c2[2] > 0:
    #             s0 = viewport[indices[0]][:2]
    #             s1 = viewport[indices[1]][:2]
    #             s2 = viewport[indices[2]][:2]
    #
    #             if not self.face_culling or self.front_facing(c0, c1, c2):
    #                 if self.on_surface(width, height, s0) and self.on_surface(width, height, s1) and self.on_surface(width, height, s2):
    #                     # pygame.draw.polygon(surface, (255, 0, 0), [s0, s1, s2])
    #                     pygame.gfxdraw.filled_trigon(surface, int(s0[0]), int(s0[1]), int(s1[0]), int(s1[1]), int(s2[0]), int(s2[1]), (255, 0, 0))


CAMERA_SPEED = 5
LOOK_SPEED = np.pi * 1
TIME_LOG_PATH = "time.pickle"

if __name__ == '__main__':
    try:
        with open(TIME_LOG_PATH, "rb") as f:
            runs = pickle.load(f)
    except:
        runs = []

    screen = pygame.display.set_mode((500, 500))
    font = pygame.freetype.SysFont("Segoe UI", 24)

    camera = Camera(1000.0, 0.5, aspect=screen.get_width() / screen.get_height())
    # camera.position = np.array([0, 0, 10], dtype=np.float64)
    # print(camera.model_matrix, camera.view_matrix)
    # camera._view_matrix = compose_matrix(np.array([0, 0, -10]), np.ones((3,)), np.identity(3))
    renderer = Renderer()
    # renderer.face_culling = False

    rotation_matrix = compose_matrix(np.ones((3,)), np.ones((3,)), rotation_matrix_z(math.pi * 0.5))
    vertex_buffer = np.array([
        [-1, -1, -1],  # bottom left back 0
        [1, -1, -1],  # bottom right back 1
        [-1, 1, -1],  # top left back 2
        [1, 1, -1],  # top right back 3
        [-1, -1, 1],  # bottom left front 4
        [1, -1, 1],  # bottom right front 5
        [-1, 1, 1],  # top left front 6
        [1, 1, 1],  # top right front 7
    ]).T
    index_buffer = np.array([
        # front
        0, 2, 1,
        2, 3, 1,
        # back
        4, 5, 6,
        6, 5, 7,
        # bottom
        0, 5, 4,
        0, 1, 5,
        # top
        2, 6, 7,
        2, 7, 3,
        # left
        0, 4, 2,
        2, 4, 6,
        # right
        1, 3, 5,
        3, 7, 5,
    ])
    # colour_buffer = np.array([index_buffer[i: i + 3] / vertex_buffer.shape[1] * 255 for i in range(0, len(index_buffer), 3)])
    colour_buffer = np.array([
        # front
        [255, 0, 0],
        [255, 0, 0],
        # back
        [0, 255, 0],
        [0, 255, 0],
        # bottom
        [0, 0, 255],
        [0, 0, 255],
        # top
        [255, 255, 0],
        [255, 255, 0],
        # left
        [255, 0, 255],
        [255, 0, 255],
        # right
        [0, 255, 255],
        [0, 255, 255],
    ])

    geometry = Geometry()
    geometry.vertex_buffer = vertex_buffer
    geometry.index_buffer = index_buffer
    geometry.colour_buffer = colour_buffer
    material = None
    mesh = Mesh(geometry, material)


    def make_cube(x, y, z):
        cube = Mesh(geometry, material)
        cube.position = np.array([x, y, z])
        return cube


    side_count = 5
    slice_count = side_count * side_count
    scale_factor = 10
    offset = -(side_count - 1) * scale_factor / 2
    # print(side_count * scale_factor)
    # print(offset)

    meshes = [make_cube(
        i // slice_count * scale_factor + offset,
        (i % slice_count) // side_count * scale_factor + offset,
        (i % slice_count) % side_count * scale_factor + offset,
    ) for i in range(side_count * side_count * side_count)]
    scene = Scene()
    scene.children = meshes
    # scene.children.append(mesh)

    # world = mesh.model_matrix.dot(vertex_buffer)
    # print(world)
    # view = camera.view_matrix.dot(world)
    # print(view)
    # clip = camera.projection_matrix.dot(view)
    # print(clip)

    clock = pygame.time.Clock()

    delta_times = []

    start_time = time.time()
    run_time = 5

    frame_count = 0

    while time.time() - start_time < run_time:
        frame_count += 1

        delta_time = clock.tick() / 1000
        t = pygame.time.get_ticks() / 1000

        events = pygame.event.get()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            camera.position += camera.front * CAMERA_SPEED * delta_time
        if keys[pygame.K_s]:
            camera.position += camera.front * -CAMERA_SPEED * delta_time
        if keys[pygame.K_a]:
            camera.position += camera.right * -CAMERA_SPEED * delta_time
        if keys[pygame.K_d]:
            camera.position += camera.right * CAMERA_SPEED * delta_time
        if keys[pygame.K_LEFT]:
            camera.rotation = camera.rotation.dot(rotation_matrix_y(LOOK_SPEED * delta_time))
        if keys[pygame.K_RIGHT]:
            camera.rotation = camera.rotation.dot(rotation_matrix_y(-LOOK_SPEED * delta_time))
        if keys[pygame.K_UP]:
            camera.rotation = camera.rotation.dot(rotation_matrix_x(-LOOK_SPEED * delta_time))
        if keys[pygame.K_DOWN]:
            camera.rotation = camera.rotation.dot(rotation_matrix_x(LOOK_SPEED * delta_time))

        # mesh.rotation = rotation_matrix_x(t).dot(rotation_matrix_y(t).dot(rotation_matrix_z(t)))
        # mesh.scale = np.array([np.sin(t) + 2, np.sin(t) + 2, np.sin(t) + 2])
        # mesh.position = np.array([np.sin(t) * 10, 0, 0])

        screen.fill((255, 255, 255))

        # renderer.draw_points(camera, mesh, screen)
        # renderer.draw_triangles(camera, mesh, screen)
        #
        # for mesh in meshes:
        #     renderer.draw_points(camera, mesh, screen)
        #     renderer.draw_triangles(camera, mesh, screen)

        mesh_count = renderer.draw_scene(scene, camera, screen)

        if delta_time != 0:
            delta_times.append(delta_time)

            if len(delta_times) > 256:
                del delta_times[0]

            font.render_to(screen, (10, 10), f"{1 / delta_time:.1f}", (0, 0, 0))
            font.render_to(screen, (10, 50), f"{1 / (sum(delta_times) / len(delta_times)):.1f}", (0, 0, 0))

        font.render_to(screen, (10, 90), f"{time.time() - start_time:.1f}", (0, 0, 0))
        font.render_to(screen, (10, 130), f"{mesh_count}", (0, 0, 0))

        pygame.display.flip()

    average_fps = frame_count / run_time

    with open("main.py", "rb") as f:
        program_hash = hashlib.sha256(f.read()).hexdigest()

    run_stats = (average_fps, run_time, frame_count, program_hash)

    runs.append(run_stats)

    with open(TIME_LOG_PATH, "wb") as f:
        pickle.dump(runs, f)

    program_stats = list(map(lambda x: x[0], filter(lambda x: x[3] == program_hash, runs)))

    print("all runs:")
    print(runs)

    print("this run:")
    print(average_fps)
    print(run_stats)

    print("averaged runs:")
    print(sum(program_stats) / len(program_stats))
