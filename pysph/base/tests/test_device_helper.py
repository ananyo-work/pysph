from unittest import TestCase
import pytest
import numpy as np

from pysph.base.utils import get_particle_array  # noqa: E402
from pysph.base.device_helper import DeviceHelper
from pysph.cpy.config import get_config
import pysph.cpy.array as array


def setup_module():
    get_config().use_openmp=True


def teardown_module():
    get_config().use_openmp=False


test_all_backends = pytest.mark.parametrize('backend',
        ['cython', 'opencl', 'cuda'])


test_fail_on_cuda = pytest.mark.parametrize('backend',
        ['cython', 'opencl', pytest.param('cuda',
            marks=pytest.mark.xfail(raises=NotImplementedError,
                reason='Scan not supported by CUDA'))])


class TestDeviceHelper(object):
    def setup(self):
        self.pa = get_particle_array(name='f', x=[0.0, 1.0], m=1.0, rho=2.0,
                                     backend='cpu')

    @test_all_backends
    def test_simple(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)

        # Then
        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    @test_all_backends
    def test_push_correctly_sets_values_with_args(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)
        assert pa.tag[0] == 0

        # When
        pa.set_device_helper(h)
        pa.x[:] = [2.0, 3.0]
        pa.rho[0] = 1.0
        pa.tag[:] = 1
        h.push('x', 'rho', 'tag')

        # Then
        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    @test_all_backends
    def test_push_correctly_sets_values_with_no_args(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        pa.x[:] = 1.0
        pa.rho[:] = 1.0
        pa.m[:] = 1.0
        pa.tag[:] = [1, 2]
        h.push()

        # Then
        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    @test_all_backends
    def test_pull_correctly_sets_values_with_args(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)
        assert pa.tag[0] == 0

        # When
        pa.set_device_helper(h)
        h.x.set(np.array([2.0, 3.0], h.x.dtype))
        h.rho[0] = 1.0
        h.tag[:] = 1
        h.pull('x', 'rho', 'tag')

        # Then
        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    @test_all_backends
    def test_pull_correctly_sets_values_with_no_args(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        h.x[:] = 1.0
        h.rho[:] = 1.0
        h.m[:] = 1.0
        h.tag[:] = np.array([1, 2], h.tag.dtype)
        h.pull()

        # Then
        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    @test_all_backends
    def test_max_provides_maximum(self, backend):
        self.setup()
        # Given/When
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # Then
        assert h.max('x') == 1.0

    @test_all_backends
    def test_that_adding_removing_prop_to_array_updates_gpu(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        pa.add_property('test', data=[3.0, 4.0])

        # Then
        assert np.allclose(pa.test, h.test.get())

        # When
        pa.remove_property('test')

        # Then
        assert not hasattr(h, 'test')
        assert not 'test' in h._data
        assert not 'test' in h.properties

    @test_all_backends
    def test_resize_works(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        pa.extend(2)
        pa.align_particles()
        h.resize(4)

        # Then
        assert np.allclose(pa.x[:2], h.x[:2].get())
        assert np.allclose(pa.m[:2], h.m[:2].get())
        assert np.allclose(pa.rho[:2], h.rho[:2].get())
        assert np.allclose(pa.tag[:2], h.tag[:2].get())

        # When
        pa.remove_particles([2, 3])
        pa.align_particles()
        old_x = h.x.data
        h.resize(2)

        # Then
        assert old_x == h.x.data
        assert np.allclose(pa.x, h.x.get())
        assert np.allclose(pa.y, h.y.get())
        assert np.allclose(pa.m, h.m.get())
        assert np.allclose(pa.rho, h.rho.get())
        assert np.allclose(pa.tag, h.tag.get())

    @test_fail_on_cuda
    def test_get_number_of_particles(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))
        h.tag.set(np.array([0, 0, 1, 0, 1], h.tag.dtype))

        h.align_particles()

        # Then
        assert h.get_number_of_particles() == 5
        assert h.get_number_of_particles(real=True) == 3

    @test_all_backends
    def test_align(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))

        indices = array.arange(4, -1, -1, dtype=np.int32,
                               backend=backend)

        h.align(indices)

        # Then
        assert np.all(h.x.get() == np.array([6., 5., 4., 3., 2.]))

    @test_fail_on_cuda
    def test_align_particles(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))
        h.tag.set(np.array([0, 0, 1, 0, 1], h.tag.dtype))

        h.align_particles()

        # Then
        x = h.x.get()
        assert np.all(np.sort(x[:-2]) == np.array([2., 3., 5.]))

    @test_fail_on_cuda
    def test_remove_particles(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        h.resize(4)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0], h.x.dtype))

        indices = np.array([1, 2], dtype=np.uint32)
        indices = array.to_device(indices, backend=backend)

        h.remove_particles(indices)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([2., 5.]))

    @test_fail_on_cuda
    def test_remove_tagged_particles(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        h.resize(5)
        h.x.set(np.array([2.0, 3.0, 4.0, 5.0, 6.0], h.x.dtype))
        h.tag.set(np.array([0, 0, 1, 0, 1], h.tag.dtype))

        h.remove_tagged_particles(1)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([2., 3., 5.]))

    @test_fail_on_cuda
    def test_add_particles(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)
        x = array.zeros(4, np.float32, backend=backend)

        h.add_particles(x=x)

        # Then
        assert np.all(np.sort(h.x.get()) == np.array([0., 0., 0., 0., 0., 1.]))

    @test_all_backends
    def test_extend(self, backend):
        self.setup()
        # Given
        pa = self.pa
        h = DeviceHelper(pa, backend=backend)

        # When
        pa.set_device_helper(h)

        h.extend(4)

        # Then
        assert h.get_number_of_particles() == 6

    @test_fail_on_cuda
    def test_append_parray(self, backend):
        self.setup()
        # Given
        pa1 = self.pa
        pa2 = get_particle_array(name='s', x=[0.0, 1.0], m=1.0, rho=2.0)
        h = DeviceHelper(pa1, backend=backend)
        pa1.set_device_helper(h)

        # When
        h.append_parray(pa2)

        # Then
        assert h.get_number_of_particles() == 4

    @test_fail_on_cuda
    def test_extract_particles(self, backend):
        self.setup()
        # Given
        pa = get_particle_array(name='f', x=[0.0, 1.0, 2.0, 3.0],
                                m=1.0, rho=2.0)
        h = DeviceHelper(pa, backend=backend)
        pa.set_device_helper(h)

        # When
        indices = np.array([1, 2], dtype=np.uint32)
        indices = array.to_device(indices, backend=backend)

        result_pa = h.extract_particles(indices)

        # Then
        assert result_pa.gpu.get_number_of_particles() == 2

