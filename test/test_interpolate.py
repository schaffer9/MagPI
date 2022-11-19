from pinns.domain import Hypercube
from pinns.interpolate import shape_function

from . import *

from scipy.stats.qmc import Sobol


class TestInterpolate(JaxTestCase):
    def test_interpolate_cube(self):
        domain = Hypercube((-1, -1, -1), (1, 1, 1))
        x_bnd = array(Sobol(2, seed=0).random_base2(6))
        x_bnd = domain.transform_bnd(x_bnd)
        x1 = [x_bnd[x_bnd[:, i] == -1, :] for i in range(3)]
        x2 = [x_bnd[x_bnd[:, i] == 1, :] for i in range(3)]
        l = shape_function(x1, x2)
        y = vmap(l)(x_bnd)
        y_true = zeros(len(x_bnd))
        self.assertIsclose(y, y_true)
