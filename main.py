import math

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


class Geometry:
    def __init__(self):
        self.vertex_buffer = np.empty((0,))
        self.index_buffer = np.empty((0,))
        self.colour_buffer = np.empty((0,))


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


class Renderer:
    def __init__(self):
        pass

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

    def transform(self, camera: Camera, game_object: Mesh, surface: pygame.Surface):
        clip = self.apply_transforms(
            np.pad(game_object.geometry.vertex_buffer, [(0, 1), (0, 0)], mode="constant", constant_values=1),
            game_object.model_matrix,
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
                if np.isfinite(screen).all():
                    # print(screen)
                    pygame.draw.circle(surface, (255, 0, 0), screen, radius)

    def draw_triangles(self, camera: Camera, mesh: Mesh, surface: pygame.Surface):
        clip, viewport = self.transform(camera, mesh, surface)

        for i in range(0, mesh.geometry.index_buffer.shape[0], 3):
            indices = mesh.geometry.index_buffer[i:i+3]
            # if clip[2, indices[0]] or clip[2, indices[1]] or clip[2, indices[2]]:

            screen1 = viewport[indices[0]][:2]
            screen2 = viewport[indices[1]][:2]
            screen3 = viewport[indices[2]][:2]

            pygame.draw.polygon(surface, (255, 0, 0), [screen1, screen2, screen3])


CAMERA_SPEED = 5


if __name__ == '__main__':
    screen = pygame.display.set_mode((500, 500))
    font = pygame.freetype.SysFont("Segoe UI", 24)

    camera = Camera(1.0, 3.0, 0.5)
    camera.position = np.array([0, 0, 10], dtype=np.float64)
    # print(camera.model_matrix, camera.view_matrix)
    # camera._view_matrix = compose_matrix(np.array([0, 0, -10]), np.ones((3,)), np.identity(3))
    renderer = Renderer()

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
        0, 1, 2,
        2, 3, 1,
        # back
        4, 5, 6,
        6, 7, 5,
        # bottom
        0, 4, 5,
        0, 1, 5,
        # top
        2, 6, 7,
        2, 3, 7,
        # left
        0, 4, 2,
        2, 6, 4,
        # right
        1, 5, 3,
        3, 7, 5,
    ])

    # print(vertex_buffer)

    geometry = Geometry()
    geometry.vertex_buffer = vertex_buffer
    geometry.index_buffer = index_buffer
    material = None
    mesh = Mesh(geometry, material)

    # world = mesh.model_matrix.dot(vertex_buffer)
    # print(world)
    # view = camera.view_matrix.dot(world)
    # print(view)
    # clip = camera.projection_matrix.dot(view)
    # print(clip)

    clock = pygame.time.Clock()

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

        font.render_to(screen, (10, 10), f"{1 / delta_time if delta_time != 0 else 0}", (0, 0, 0))

        pygame.display.flip()
