import math
from enum import Enum

import pygame
import numpy as np
from numpy.typing import NDArray


np.set_printoptions(suppress=True)
pygame.init()


def perspective_matrix(s: float, f: float, n: float) -> NDArray:
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)],
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


def distance_squared(a: NDArray, b: NDArray) -> NDArray:
    l0 = a[0] - b[0]
    l1 = a[1] - b[1]
    l2 = a[2] - b[2]
    return l0 * l0 + l1 * l1 + l2 * l2


def zero_distance_squared(v: NDArray) -> NDArray:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


class Geometry:
    def __init__(self):
        self._vertex_buffer: NDArray = np.empty((0,))
        self.index_buffer: NDArray = np.empty((0,))
        self.colour_buffer: NDArray = np.empty((0,))
        self.bounding_sphere: float = 0

    @property
    def vertex_buffer(self) -> NDArray:
        return self._vertex_buffer

    @vertex_buffer.setter
    def vertex_buffer(self, value: NDArray):
        if value.shape[1] == 3:  # if not already padded, one-time pre-pad for performance
            self._vertex_buffer = np.pad(value, [(0, 0), (0, 1)], mode="constant", constant_values=1)
        else:
            self._vertex_buffer = value
        self.bounding_sphere = self.compute_bounding_sphere()

    def compute_bounding_sphere(self):
        r_squared = 0
        for vertex in self.vertex_buffer:
            current_r_squared = zero_distance_squared(vertex)
            if current_r_squared > r_squared:
                r_squared = current_r_squared

        return np.sqrt(r_squared)


class Object3D:
    def __init__(self):
        self._position: NDArray = np.zeros((3,), dtype=np.float64)
        self._scale: NDArray = np.ones((3,), dtype=np.float64)
        self._rotation: NDArray = np.identity(3)
        self._model_matrix: NDArray = np.identity(4)
        self._matrix_needs_update: bool = False

    @property
    def model_matrix(self):
        if self._matrix_needs_update:
            self.update_model_matrix()
        return self._model_matrix

    @property
    def position(self) -> NDArray:
        return self._position

    @position.setter
    def position(self, value: NDArray):
        self._position = value
        self._matrix_needs_update = True

    @property
    def scale(self) -> NDArray:
        return self._scale

    @scale.setter
    def scale(self, value: NDArray):
        self._scale = value
        self._matrix_needs_update = True

    @property
    def rotation(self) -> NDArray:
        return self._rotation

    @rotation.setter
    def rotation(self, value: NDArray):
        self._rotation = value
        self._matrix_needs_update = True

    def update_model_matrix(self):
        self._matrix_needs_update = False
        self._model_matrix = compose_matrix(self._position, self._scale, self._rotation)


class Mesh(Object3D):
    def __init__(self, geometry: Geometry, material):
        super().__init__()
        self.geometry: Geometry = geometry


class Camera(Object3D):
    def __init__(self, s: float, f: float, n: float):
        super().__init__()
        self.s: float = s
        self.f: float = f
        self.n: float = n
        self.projection_matrix: NDArray = perspective_matrix(s, f, n)
        self._view_matrix: NDArray = np.identity(4)

    @property
    def view_matrix(self):
        if self._matrix_needs_update:
            self.update_model_matrix()
        return self._view_matrix

    def update_view_matrix(self):
        self._view_matrix = np.linalg.inv(self.model_matrix)

    def update_model_matrix(self):
        super().update_model_matrix()
        self.update_view_matrix()


class WindingOrder(Enum):
    CLOCKWISE = False
    ANTICLOCKWISE = True


class Renderer:
    def __init__(self):
        self.winding_order: WindingOrder = WindingOrder.ANTICLOCKWISE
        self.face_culling: bool = True

    @staticmethod
    def apply_transforms(vertex_buffer: NDArray, model_matrix: NDArray, view_matrix: NDArray, projection_matrix: NDArray) -> NDArray:
        return projection_matrix.dot(view_matrix.dot(model_matrix.dot(vertex_buffer)))

    @staticmethod
    def viewport_transform(ndc: NDArray, x: float, y: float, width: float, height: float, f: float, n: float) -> NDArray:
        half_width = width / 2
        half_height = height / 2

        # print(np.array([[half_width, half_height, (f - n) / 2]]).T)
        # print(np.array([[x + half_width, y + half_height, (f + n) / 2]]).T)

        return ndc * np.array([[half_width, half_height, (f - n) / 2]]).T + np.array([[x + half_width, y + half_height, (f + n) / 2]]).T
        # return ndc + np.array([[2, 1, 1]]).T

    @staticmethod
    def on_surface(surface: pygame.surface.Surface, point: NDArray):
        return 0 < point[0] < surface.get_width() and 0 < point[1] < surface.get_height()

    def front_facing(self, a, b, c):
        n = np.dot(a, np.cross(b - a, c - a))
        return n == 0 or n > 0 != self.winding_order

    def transform(self, camera: Camera, mesh: Mesh, surface: pygame.Surface):
        clip = self.apply_transforms(
            mesh.geometry.vertex_buffer.T,
            mesh.model_matrix,
            camera.view_matrix,
            camera.projection_matrix
        )
        # print(clip)
        # print("clip")
        # print(clip)
        # print(clip, clip[3])
        ndc = (clip / clip[3])[:3]
        # print(ndc)
        # print("ndc")
        # print(ndc)
        size = surface.get_size()
        viewport = self.viewport_transform(ndc, 0, 0, size[0], size[1], camera.f, camera.n).T

        return clip, viewport

    def draw_points(self, camera: Camera, mesh: Mesh, surface: pygame.Surface, radius: float = 2):
        # clip = self.apply_transforms(game_object.geometry.vertex_buffer, game_object.model_matrix, camera.view_matrix, camera.projection_matrix)
        # # print(clip)
        # # print("clip")
        # # print(clip)
        # # print(clip, clip[3])
        # ndc = (clip / clip[3])[:3]
        # # print(ndc)
        # # print("ndc")
        # # print(ndc)
        # size = surface.get_size()
        # viewport = self.viewport_transform(ndc, 0, 0, size[0], size[1], camera.f, camera.n).T
        # # print("viewport")
        # # print(viewport)
        #
        # # print(viewport)

        clip, viewport = self.transform(camera, mesh, surface)

        for r in range(viewport.shape[0]):
            # print(vertex)
            vertex = viewport[r]
            if clip[2, r] > 0:  # in front of camera
                screen = vertex[:2]
                pygame.draw.circle(surface, (255, 0, 0), screen, radius)
                # if np.isfinite(screen).all():
                #     # print(screen)


    def draw_triangles(self, camera: Camera, mesh: Mesh, surface: pygame.Surface):
        clip, viewport = self.transform(camera, mesh, surface)

        for i in range(0, mesh.geometry.index_buffer.shape[0], 3):
            indices = mesh.geometry.index_buffer[i:i + 3]

            c0 = clip[:3, indices[0]].T
            c1 = clip[:3, indices[1]].T
            c2 = clip[:3, indices[2]].T

            # if clip[2, indices[0]] > 0 or clip[2, indices[1]] > 0 or clip[2, indices[2]] > 0:
            if c0[2] > 0 or c1[2] > 0 or c2[2] > 0:
                s0 = viewport[indices[0]][:2]
                s1 = viewport[indices[1]][:2]
                s2 = viewport[indices[2]][:2]

                if not self.face_culling or self.front_facing(c0, c1, c2):
                    if self.on_surface(surface, s0) and self.on_surface(surface, s1) and self.on_surface(surface, s2):
                        pygame.draw.polygon(surface, (255, 0, 0), [s0, s1, s2])


CAMERA_SPEED = 5


if __name__ == '__main__':
    screen = pygame.display.set_mode((500, 500))
    font = pygame.freetype.SysFont("Segoe UI", 24)

    camera = Camera(1.0, 3.0, 0.5)
    camera.position = np.array([0, 0, 10], dtype=np.float64)
    # print(camera.model_matrix, camera.view_matrix)
    # camera._view_matrix = compose_matrix(np.array([0, 0, -10]), np.ones((3,)), np.identity(3))
    renderer = Renderer()
    renderer.face_culling = False

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
    ])
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

    # print(vertex_buffer)

    geometry = Geometry()
    geometry.vertex_buffer = vertex_buffer
    geometry.index_buffer = index_buffer
    material = None
    mesh = Mesh(geometry, material)
    meshes = [Mesh(geometry, material) for i in range(100)]

    # world = mesh.model_matrix.dot(vertex_buffer)
    # print(world)
    # view = camera.view_matrix.dot(world)
    # print(view)
    # clip = camera.projection_matrix.dot(view)
    # print(clip)

    clock = pygame.time.Clock()

    delta_times = []

    while True:
        delta_time = clock.tick() / 1000
        time = pygame.time.get_ticks() / 1000

        events = pygame.event.get()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            camera.position += np.array([0, 0, -CAMERA_SPEED * delta_time])
        if keys[pygame.K_s]:
            camera.position += np.array([0, 0, CAMERA_SPEED * delta_time])
        if keys[pygame.K_a]:
            camera.position += np.array([-CAMERA_SPEED * delta_time, 0, 0])
        if keys[pygame.K_d]:
            camera.position += np.array([CAMERA_SPEED * delta_time, 0, 0])

        mesh.rotation = rotation_matrix_x(time).dot(rotation_matrix_y(time).dot(rotation_matrix_z(time)))
        mesh.scale = np.array([np.sin(time) + 2, np.sin(time) + 2, np.sin(time) + 2])
        mesh.position = np.array([np.sin(time) * 10, 0, 0])

        screen.fill((255, 255, 255))

        renderer.draw_points(camera, mesh, screen)
        renderer.draw_triangles(camera, mesh, screen)

        for mesh in meshes:
            renderer.draw_points(camera, mesh, screen)
            renderer.draw_triangles(camera, mesh, screen)

        if delta_time != 0:
            delta_times.append(delta_time)

            if len(delta_times) > 256:
                del delta_times[0]

            font.render_to(screen, (10, 10), f"{1 / delta_time:.1f}", (0, 0, 0))
            font.render_to(screen, (10, 50), f"{1 / (sum(delta_times) / len(delta_times)):.1f}", (0, 0, 0))

        pygame.display.flip()
