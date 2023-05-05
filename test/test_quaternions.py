from . import *

from pinns.quaternions import (
    Quaternion,
    sgn,
    quanternion_exp,
    from_axis_angle,
    from_euler_angles,
    quaternion_rotation,
)


class TestQuaternion(JaxTestCase):
    def test_000_init(self):
        q = Quaternion(1.0, (0.0, 0.0, 0.0))
        self.assertIsclose(q.real, 1.0)
        self.assertIsclose(q.imag, array((0.0, 0.0, 0.0)))

    def test_001_add(self):
        q1 = Quaternion(1.0, (0.0, 0.0, 0.0))
        q2 = Quaternion(0.0, (1.0, 1.0, 1.0))
        q = q1 + q2
        self.assertIsclose(q.real, 1.0)
        self.assertIsclose(q.imag, array((1.0, 1.0, 1.0)))

    def test_002_add(self):
        q = Quaternion(1.0, (0.0, 0.0, 0.0)) + 1
        self.assertIsclose(q.real, 2.0)
        self.assertIsclose(q.imag, array((1.0, 1.0, 1.0)))

    def test_003_sub(self):
        q1 = Quaternion(1.0, (0.0, 0.0, 0.0))
        q2 = Quaternion(0.0, (-1.0, -1.0, -1.0))
        q = q1 - q2
        self.assertIsclose(q.real, 1.0)
        self.assertIsclose(q.imag, array((1.0, 1.0, 1.0)))

    def test_004_sub(self):
        q = Quaternion(1.0, (0.0, 0.0, 0.0)) - 1
        self.assertIsclose(q.real, 0.0)
        self.assertIsclose(q.imag, array((-1.0, -1.0, -1.0)))

    def test_005_mul(self):
        q1 = Quaternion(0.0, (1.0, 0.0, 0.0))
        q2 = Quaternion(0.0, (1.0, 0.0, 0.0))
        q = q1 * q2
        self.assertIsclose(q.real, -1.0)
        self.assertIsclose(q.imag, array((0.0, 0.0, 0.0)))

    def test_006_mul(self):
        q1 = Quaternion(0.0, (1.0, 0.0, 0.0))
        q2 = Quaternion(0.0, (0.0, 1.0, 0.0))
        q = q1 * q2
        self.assertIsclose(q.real, 0.0)
        self.assertIsclose(q.imag, array((0.0, 0.0, 1.0)))

    def test_007_mul(self):
        q1 = Quaternion(0.0, (1.0, 0.0, 0.0))
        q2 = Quaternion(0.0, (0.0, 0.0, -1.0))
        q = q1 * q2
        self.assertIsclose(q.real, 0.0)
        self.assertIsclose(q.imag, array((0.0, 1.0, 0.0)))

    def test_008_mul(self):
        q1 = Quaternion(1.0, (1.0, 0.0, 0.0))
        q2 = Quaternion(1.0, (1.0, 0.0, 0.0))
        q = q1 * q2
        self.assertIsclose(q.real, 0.0)
        self.assertIsclose(q.imag, array((2.0, 0.0, 0.0)))

    def test_009_mul_scalar(self):
        q = Quaternion(1.0, (2.0, 2.0, 2.0)) * 2
        self.assertIsclose(q.real, 2.0)
        self.assertIsclose(q.imag, array((4.0, 4.0, 4.0)))

    def test_010_resciprocal(self):
        q = Quaternion(1.0, (2.0, 2.0, 2.0))
        self.assertIsclose(abs(q * q.reciprocal()), 1.0)

    def test_011_conj(self):
        q = Quaternion(1.0, (2.0, 2.0, 2.0))
        qconj = Quaternion(1.0, (-2.0, -2.0, -2.0))
        self.assertEqual(q.conj(), qconj)

    def test_012_sgn(self):
        q = array((0.0, 0.0, 0.0))
        self.assertIsclose(sgn(q), q)

    def test_013_sgn(self):
        q = array((1.0, 0.0, 0.0))
        self.assertIsclose(sgn(q), q)

    def test_014_sgn(self):
        q = array((0.0, 0.0, 1.0))
        self.assertIsclose(sgn(q), q)

    def test_015_pow(self):
        q = Quaternion(1.0, (2.0, 2.0, 2.0))
        q2 = q**2
        qq = q * q
        self.assertIsclose(q2.real, qq.real)
        self.assertIsclose(q2.imag, qq.imag)

    def test_016_pow(self):
        q = Quaternion(1.0, (3.0, 5.0, 6.0))
        q3 = q**3
        qqq = q * q * q
        self.assertIsclose(q3.real, qqq.real)
        self.assertIsclose(q3.imag, qqq.imag)

    def test_017_rpow(self):
        q = Quaternion(1.0, (1.0, 0.0, 0.0))
        eq = jnp.e**q
        p = quanternion_exp(q)
        self.assertIsclose(eq.real, p.real)
        self.assertIsclose(eq.imag, p.imag)

    def test_018_rotate_point(self):
        p = array([1.0, 0.0, 0.0])

        q = from_axis_angle(pi / 2, (1.0, 0.0, 0.0))
        y = quaternion_rotation(p, q)
        self.assertIsclose(y, array((1.0, 0.0, 0.0)))

        q = from_axis_angle(pi / 2, (0.0, 1.0, 0.0))
        y = quaternion_rotation(p, q)
        self.assertIsclose(y, array((0.0, 0.0, 1.0)))

        q = from_axis_angle(pi / 2, (0.0, 0.0, 1.0))
        y = quaternion_rotation(p, q)
        self.assertIsclose(y, array((0.0, -1.0, 0.0)))

    def test_019_rotate_euler_angles(self):
        p = array([1.0, 0.0, 0.0])
        q = from_euler_angles((pi / 2.0, 0.0, 0.0))
        y = quaternion_rotation(p, q)
        self.assertIsclose(y, array((1.0, 0.0, 0.0)))

        q = from_euler_angles((0.0, pi / 2, 0.0))
        y = quaternion_rotation(p, q)
        self.assertIsclose(y, array((0.0, 0.0, 1.0)))

        q = from_euler_angles((0.0, 0.0, pi / 2))
        y = quaternion_rotation(p, q)
        self.assertIsclose(y, array((0.0, -1.0, 0.0)))
