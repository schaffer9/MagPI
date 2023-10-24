from magpi import calc

from . import *


# class TestCross(JaxTestCase):
#     def test_cross2d(self):
#         a = array([[1, 0], [0, 1]])
#         b = array([[0, 1], [1, 0]])
#         self.assertIsclose(jit(vmap(calc.cross))(a, b), array([1, -1]))

#     def test_cross3d(self):
#         a = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         b = array([[0, 1, 0], [1, 0, 0], [1, 1, 0]])
#         self.assertIsclose(vmap(calc.cross)(a, b), array([[0, 0, 1], [0, 0, -1], [-1, 1, 0]]))


class TestDot(JaxTestCase):
    def test_dot(self):
        a = array([1, 1, 2])
        b = array([1, 1, 2])
        self.assertIsclose(calc.dot(a, b), array(6))


class TestDerivative(JaxTestCase):
    def test_01_derivative_of_simple_function(self):
        g = lambda x: sin(x)
        dg = jit(calc.derivative(g))(0.5)
        self.assertIsclose(dg, cos(0.5))

    def test_02_derivative_of_function_with_multiple_inputs(self):
        g = lambda x, y: sin(x) * cos(y)
        dg = calc.derivative(1)(g)(0.5, 1.0)
        self.assertIsclose(dg, -sin(0.5) * sin(1.0))

    def test_03_derivative_of_function_with_multiple_inputs(self):
        g = lambda x, y: sin(x) * cos(y)
        dg = calc.derivative([1, 0])(g)(0.5, 1.0)
        self.assertIsclose(dg, -cos(0.5) * sin(1.0))

    def test_04_derivative_with_named_variables(self):
        g = lambda x, y: sin(x) * cos(y)
        dg = jit(calc.derivative(["x", "y", "y"])(g))(0.5, 1.0)
        self.assertIsclose(dg, -cos(0.5) * cos(1.0))

    def test_05_directional_derivative(self):
        g = lambda x, y: sin(x) * cos(y)
        dg = jit(calc.derivative([[0.0, 1.0]])(g))(0.5, 1.0)
        self.assertIsclose(dg, -sin(0.5) * sin(1.0))

    def test_06_vector_input(self):
        g = lambda x: sin(x[0]) * cos(x[1])
        dg = jit(calc.derivative([1], argnums=0)(g))(array([0.5, 1.0]))
        self.assertIsclose(dg, -sin(0.5) * sin(1.0))

    def test_07_derivative_of_simple_function_with_argnums(self):
        g = lambda x, y: sin(x * y**2)
        dg = jit(calc.derivative(g, 1))(1.0, 0.5)
        self.assertIsclose(dg, cos(0.5**2))


class TestDiv(JaxTestCase):
    def test_01_div(self):
        def f(x):
            return x**2

        x = array([2.0, 2.0])
        div = calc.divergence(f)(x)
        self.assertIsclose(div, array(8.0))
        
    def test_02_div_batch(self):
        def f(x):
            return asarray([x**2, x])

        x = array([2.0, 2.0])
        div = calc.divergence(f)(x)
        self.assertIsclose(div, array([8.0, 2.0]))
        
    def test_03_value_and_div(self):
        def f(x):
            return asarray([x**2, x])

        x = array([2.0, 2.0])
        y, div = calc.value_and_divergence(f)(x)
        self.assertIsclose(div, array([8.0, 2.0]))
        self.assertIsclose(y, array([[4.0, 4.0], [2.0, 2.0]]))
        
    def test_04_value_and_div_with_pytree(self):
        def f(x):
            return asarray([x**2, x]), asarray([x**2, x])

        x = array([2.0, 2.0])
        y, div = calc.value_and_divergence(f)(x)
        print(div)
        self.assertIsclose(div, (array([8.0, 2.0]), array([8.0, 2.0])))
        self.assertIsclose(y, (array([[4.0, 4.0], [2.0, 2.0]]), array([[4.0, 4.0], [2.0, 2.0]])))
        

class TestCurl(JaxTestCase):
    def test_curl2d(self):
        def f(x):
            o = stack([x[1] ** 2, x[0] ** 2])
            return o

        x = array([[2.0, 1.0], [2.0, 1.0]])
        curl = vmap(calc.curl(f))(x)
        self.assertIsclose(curl, array([2.0, 2.0]))

    def test_curl3d(self):
        def f(x):
            o = stack([x[1] ** 2 * x[2], x[0] * x[2] ** 2, x[0] ** 2 * x[1]])
            return o

        x = array([[2.0, 1.0, 2.0], [2.0, 1.0, 2.0]])
        curl = vmap(calc.curl(f))(x)
        self.assertIsclose(curl, array([[-4, -3, 0.0], [-4, -3, 0.0]]))


class TestLaplace(JaxTestCase):
    def test_laplace(self):
        def f(x):
            o = x[0] ** 2 + x[1] ** 2 + x[2] ** 3
            return o

        x = array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
        lap = vmap(calc.laplace(f))(x)
        self.assertIsclose(lap, array([16.0, 16.0]))

    def test_laplace_on_vector_valued_function(self):
        """This should give the laplace of each
        individual output of the function
        """

        def f(x):
            o = x[0] ** 2 + x[1] ** 2 + x[2] ** 3
            return stack([o, 2 * o], axis=-1)

        x = array([1.0, 1.0, 2.0])
        lap = calc.laplace(f)(x)
        self.assertIsclose(lap, array([16.0, 32.0]))

    def test_laplace_on_matrix_function(self):
        def f(x):
            o = x[0] ** 2 + x[1] ** 2 + x[2] ** 3
            return array([[o, 2 * o], [o, 3 * o]])

        x = array([1.0, 1.0, 2.0])
        lap = calc.laplace(f)(x)
        self.assertIsclose(lap, array([[16.0, 32.0], [16.0, 48]]))

    def test_laplace_on_pytree(self):
        def f(x):
            o = x[0] ** 2 + x[1] ** 2 + x[2] ** 3
            return o, 2 * o

        x = array([1.0, 1.0, 2.0])
        lap = calc.laplace(f)(x)
        self.assertTrue(isinstance(lap, tuple))
        self.assertIsclose(lap[0], 16.0)
        self.assertIsclose(lap[1], 32.0)

    def test_laplace_on_scalar_input_and_scalar_output(self):
        def f(x):
            return x**3

        x = 1.0
        lap = calc.laplace(f)(x)
        self.assertEqual(lap.shape, ())
        self.assertIsclose(lap, 6)


class TestValueAndJacfwd(JaxTestCase):
    def test_00_sin(self):
        y, dy = calc.value_and_jacfwd(sin)(pi / 2)
        self.assertIsclose(y, 1.0)
        self.assertIsclose(dy, 0.0)

    def test_01_2d_with_aux(self):
        def f(x1, x2):
            return x1**2 + x2, x2

        (y, aux), dy = calc.value_and_jacfwd(f, argnums=(0, 1), has_aux=True)(
            array([1.0, 1.0]), 2.0
        )

        self.assertIsclose(y, array([3.0, 3.0]))
        self.assertIsclose(aux, 2.0)
        self.assertIsclose(dy[0], array([[2.0, 0.0], [0.0, 2.0]]))
        self.assertIsclose(dy[1], array([1.0, 1.0]))

    def test_02_has_aux(self):
        def f(x):
            return sin(x), {"bar": 2}

        (_, aux), _ = calc.value_and_jacfwd(f, has_aux=True)(pi / 2)
        self.assertEqual(aux["bar"], 2)

    def test_03_pytree_output(self):
        def f(x):
            return sin(x), {"bar": 2}

        y, dy = calc.value_and_jacfwd(f)(pi / 2)
        self.assertIsclose(y[0], 1.0)
        self.assertIsclose(y[1]["bar"], 2.0)
        self.assertIsclose(dy[0], 0.0)
        self.assertIsclose(dy[1]["bar"], 0.0)


class TestValueAndJacrev(JaxTestCase):
    def test_00_sin(self):
        y, dy = calc.value_and_jacrev(sin)(pi / 2)
        self.assertIsclose(y, 1.0)
        self.assertIsclose(dy, 0.0)

    def test_01_2d_with_aux(self):
        def f(x1, x2):
            return x1**2 + x2, x2

        (y, aux), dy = calc.value_and_jacrev(f, argnums=(0, 1), has_aux=True)(
            array([1.0, 1.0]), 2.0
        )

        self.assertIsclose(y, array([3.0, 3.0]))
        self.assertIsclose(aux, 2.0)
        self.assertIsclose(dy[0], array([[2.0, 0.0], [0.0, 2.0]]))
        self.assertIsclose(dy[1], array([1.0, 1.0]))

    def test_02_has_aux(self):
        def f(x):
            return sin(x), {"bar": 2}

        (_, aux), _ = calc.value_and_jacrev(f, has_aux=True)(pi / 2)
        self.assertEqual(aux["bar"], 2)

    def test_03_pytree_output(self):
        def f(x):
            return sin(x), {"bar": 2.0}

        y, dy = calc.value_and_jacrev(f)(pi / 2)
        self.assertIsclose(y[0], 1.0)
        self.assertIsclose(y[1]["bar"], 2.0)
        self.assertIsclose(dy[0], 0.0)
        self.assertIsclose(dy[1]["bar"], 0.0)
