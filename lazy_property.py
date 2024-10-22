from typing import Callable, List, TypeVar, Generic, Hashable
import weakref

T = TypeVar("T")


class LazyProperty(Generic[T]):
    def __init__(self, instance_dependencies: List[Callable], compute: Callable[..., T] = None):
        self.compute: Callable[..., T] = compute
        self.instance_dependencies: List[Callable] = instance_dependencies
        self.dependency_hashes: List[int | weakref.ref] = []
        self._first_compute: bool = True
        self._value: T = None
        self._name: str = ""

    def compute_dependencies(self, instance) -> List:
        return list(map(lambda x: x(instance), self.instance_dependencies))

    # slow old method
    # @staticmethod
    # def hash_dependencies(instance_dependencies: List) -> bytes:
    #     sha256 = hashlib.sha256()
    #     for dependency in instance_dependencies:
    #         hash_component = hash(dependency)
    #         sha256.update(hash_component.to_bytes(hash_component.bit_length(), "big", signed=True))
    #     return sha256.digest()

    @staticmethod
    def hash_dependencies(instance_dependencies: List) -> List[int | weakref.ref]:
        return [hash(dep) if isinstance(dep, Hashable) else weakref.ref(dep) for dep in instance_dependencies]

    @staticmethod
    def compare_id(a: int | weakref.ref, b: int | weakref.ref) -> bool:
        if isinstance(a, weakref.ref):
            return a is b
        return a == b

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner):
        instance_dependencies = self.compute_dependencies(instance)
        dependency_hashes = self.hash_dependencies(instance_dependencies)
        # print(dependency_hashes, self.dependency_hashes)
        if self._first_compute or any(not self.compare_id(old, new) for old, new in zip(self.dependency_hashes, dependency_hashes)):
            self._first_compute = False
            self.dependency_hashes = dependency_hashes
            self._value = self.compute(instance, *instance_dependencies)
        return self._value

    def __set__(self, instance, value):
        raise AttributeError(f"{self.__class__.__name__} {self._name} cannot be set.")

    def __delete__(self, instance):
        raise AttributeError(f"{self.__class__.__name__} {self._name} cannot be deleted.")

    def __call__(self, compute: Callable[..., T]):
        self.compute = compute
        return self


def lazy_property(instance_dependencies: List[Callable]):
    prop = LazyProperty(instance_dependencies=instance_dependencies)

    def lazy_property_wrapper(compute: Callable[..., T]):
        prop.compute = compute
        return prop

    return lazy_property_wrapper
