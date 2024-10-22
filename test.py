import timeit
import numpy as np
from numpy.typing import NDArray


# def column_v4_to_v3(vector: NDArray) -> NDArray:
#     return np.array([vector[0, 0], vector[1, 0], vector[2, 0]])
#
#
# v = np.array([[0], [0], [-1], [1]], dtype=np.float64)
#
v = np.array([0, 0, -1], dtype=np.float64)
number = 100000
#
# print(timeit.timeit(lambda: column_v4_to_v3(v), number=number))
# print(timeit.timeit(lambda: v.flatten(), number=number))
#
# print(timeit.timeit(lambda: v.flatten()[:3], number=number))
# print(timeit.timeit(lambda: v[:3, 0].flatten(), number=number))
#
# print(v[:3, 0].flatten(), v.flatten()[:3])


# v = np.array([0, 0, -1], dtype=np.float64)
#
#
# def v3_to_column_v4(vector: NDArray) -> NDArray:
#     array = np.empty((4, 1))
#     array[0, 0] = vector[0]
#     array[1, 0] = vector[1]
#     array[2, 0] = vector[2]
#     array[3, 0] = 1
#     return array
#
#
# def v3_to_column_v4_1(vector: NDArray) -> NDArray:
#     array = np.empty((4, 1))
#     array[:3] = vector.reshape((3, 1))
#     return array
#
#
# def v3_to_column_v4_2(vector: NDArray) -> NDArray:
#     return np.append(vector, [1]).reshape((4, 1))
#
#
# print(timeit.timeit(lambda: v3_to_column_v4(v), number=number))
# print(timeit.timeit(lambda: v3_to_column_v4_1(v), number=number))
# print(timeit.timeit(lambda: v3_to_column_v4_2(v), number=number))
#
# print(v3_to_column_v4(v))
# print(v3_to_column_v4_1(v))
# print(v3_to_column_v4_2(v))


# def magnitude_squared(v: NDArray) -> float:
#     return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
#
#
# def magnitude(v: NDArray) -> float:
#     return np.sqrt(magnitude_squared(v))
#
#
# def np_magnitude(v):
#     return np.sqrt((v**2).sum())
#
#
# def np_magnitude_linalg(v):
#     return np.linalg.norm(v)
#
#
# # print(timeit.timeit(lambda: magnitude(v), number=number))
# # print(timeit.timeit(lambda: np_magnitude(v), number=number))
# # print(timeit.timeit(lambda: np_magnitude_linalg(v), number=number))

a = np.array([0, 0, -1], dtype=np.float64)
b = np.array([1, 0, 0], dtype=np.float64)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


print(timeit.timeit(lambda: dot(a, b), number=number))
print(timeit.timeit(lambda: np.dot(a, b), number=number))

# def cross(a, b):
#     return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
#
#
# print(cross(a, b))
# print(np.cross(a, b))
#
# print(timeit.timeit(lambda: cross(a, b), number=number))
# print(timeit.timeit(lambda: np.cross(a, b), number=number))
