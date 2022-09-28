from pinns.prelude import *

Array = Any
Scalar = Array
Origin = Array
Integrand = Callable[..., Scalar]
QuadRule = Callable[[Integrand, Scalar, Scalar], Scalar]


def midpoint(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    return (b - a) * f((a + b) / 2)


def trap(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    return (b - a) / 2 * (f(a) + f(b))


def simpson(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    m = (a + b) / 2
    return (b - a) / 6 * (f(a) + 4 * f(m) + f(b))


def gauss2(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    def t(x):
        return (a + b) / 2 + x * (b - a) / 2
    return (b - a) / 2 * (f(t(-1 / sqrt(3))) + f(t(1 / sqrt(3))))


def gauss3(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    def t(x):
        return (a + b) / 2 + x * (b - a) / 2
    return (b - a) / 2 * (
        8 / 9 * f(t(0.)) + 
        5 / 9 * f(t(-sqrt(1 / 5))) + 
        5 / 9 * f(t(sqrt(1 / 5)))
    )


def gauss4(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    def t(x):
        return (a + b) / 2 + x * (b - a) / 2
    u, v = sqrt(3 / 7 - 2 / 7 * sqrt(6 / 5)), sqrt(3 / 7 + 2 / 7 * sqrt(6 / 5))
    w1, w2 = (18 + sqrt(30)) / 36, (18 - sqrt(30)) / 36
    return (b - a) / 2 * (
        w1 * f(t(u)) + 
        w1 * f(t(-u)) + 
        w2 * f(t(v)) + 
        w2 * f(t(-v))
    )


def gauss5(f: Integrand, a: Scalar, b: Scalar) -> Scalar:
    def t(x):
        return (a + b) / 2 + x * (b - a) / 2
    u = 1 / 3 * sqrt(5 - 2 * sqrt(10 / 7))
    v = 1 / 3 * sqrt(5 + 2 * sqrt(10 / 7))
    w0 = 128 / 225
    w1 = (322 + 13 * sqrt(70)) / 900
    w2 = (322 - 13 * sqrt(70)) / 900
    return (b - a) / 2 * (
        w0 * f(t(0.)) + 
        w1 * f(t(u)) + 
        w1 * f(t(-u)) + 
        w2 * f(t(v)) + 
        w2 * f(t(-v))
    )


def integrate(
    f: Integrand, 
    domain: Array | list[Array], 
    *args, 
    method: QuadRule=simpson, 
    **kwargs
) -> Scalar:
    """Integrates over the given domain with the provided quadrature
    rule. The domain can either be an array or a list of arrays.

    Examples
    --------
    This integrates $f$ over the domain $[0, 1]$
        >>> f = lambda x: 2 * x
        >>> d = array([0, 1])
        >>> float(integrate(f, d, method=midpoint))
        1.0

    Using more nodal points, composite rules are used
        >>> f = lambda x: sin(x)
        >>> d = linspace(0, pi, 30)
        >>> F = integrate(f, d)
        >>> bool(jnp.isclose(F, -cos(pi) + 1))
        True
    
    For multivariate functions, the domain can be a list indicating a
    rectangular domain. Also additional parameters can be passed using 
    ``*args`` and ``*kwargs``. In this example :math:`x_0` is integrated over
    :math:`[-1, 1]`, :math:`x_1` over :math:`[0, 1]`, :math:`a=1` and :math:`b=2`

        >>> f = lambda x, a, b: a * x[0] ** 2 + b * x[1]
        >>> d = [
        ...     linspace(-1, 1, 2),
        ...     linspace(0, 1, 2),
        ... ]
        >>> F = integrate(f, d, 1., method=gauss2, b=2.)
        >>> bool(jnp.isclose(F, 2.66666666))
        True

    Parameters
    ----------
    f : Integrand
    domain : Array | list[Array]
        nodal points for each dimension
    method : QuadRule, optional
        Quadrature rule method. The midpoint rule can be written as
        ``lambda f, a, b: (b - a) * f((a + b) / 2)``. The module provides
        ``trap, midpoint, simpson`` and ``gauss2, gauss3, gauss4, gauss5``
        for Gauss-Legendre quadrature.
        
    Returns
    -------
    Scalar
    """
    if not isinstance(domain, (list, tuple)):
        assert len(domain.shape) > 0
        assert len(domain.shape) <= 2
        if len(domain.shape) == 1:
            domain = [domain]

    def g(*int_args):
        return f(stack(int_args), *args, **kwargs)

    def _integrate(f, support):
        a = support[:-1]
        b = support[1:]
        return sum(vmap(method, (None, 0, 0))(f, a, b))

    def _int(f, s):
        return lambda *args: _integrate(partial(f, *args), s)
    
    _f = g
    for s in reversed(domain):
        _f = _int(_f, s)

    return _f()


def integrate_disk(
    f: Integrand,
    r: Scalar,
    o: Origin,
    n: int | tuple[int, int],
    *args,
    method: QuadRule=simpson,
    **kwargs
) -> Scalar:
    """Integrates over a disk of radius ``r`` and origin ``o``.

    Examples
    --------
        >>> f = lambda x: 1.
        >>> area = integrate_disk(f, 1., array([0., 0.]), 20)
        >>> bool(jnp.isclose(area, pi))
        True

    Parameters
    ----------
    f : Integrand
    r : Scalar
        radius
    o : Origin
        origin
    n : int | tuple[int, int]
        Nodes in each dimension.
    method : QuadRule, optional
        Quadrature rule, by default simpson

    Returns
    -------
    Scalar
    """
    if not isinstance(n, tuple):
        n = (n, n)
    else:
        assert len(n) == 2

    def g(u, *args, **kwargs):
        r, phi = u
        x = r * cos(phi)
        y = r * sin(phi)
        p = stack([x, y]) + o
        return f(p, *args, **kwargs) * r
    
    support = [
        jnp.linspace(0, r, n[0]),
        jnp.linspace(0, 2 * pi, n[1]),
    ]
    return integrate(g, support, *args, method=method, **kwargs)


def integrate_sphere(
    f: Integrand, 
    r: Scalar, 
    o: Origin, 
    n: int | tuple[int, int, int], 
    *args, 
    method: QuadRule=simpson, 
    **kwargs
) -> Scalar:
    """Integrates over a sphere of radius ``r`` and origin ``o``.

    Parameters
    ----------
    f : Integrand
    r : Scalar
        radius
    o : Origin
        origin
    n : int | tuple[int, int, int]
        Nodes in each dimension.
    method : QuadRule, optional
        Quadrature rule, by default simpson

    Returns
    -------
    Scalar
    """
    if not isinstance(n, tuple):
        n = (n, n, n)
    else:
        assert len(n) == 3

    def g(t: Array, *args, **kwargs) -> Scalar:
        r, phi, theta = t
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        p = stack([x, y, z]) + o
        return f(p, *args, **kwargs) * r ** 2 * sin(phi)
    
    domain = [
        jnp.linspace(0, r, n[0]),
        jnp.linspace(0, pi, n[1]),
        jnp.linspace(0, 2 * pi, n[2]),
    ]
    return integrate(g, domain, *args, method=method, **kwargs)
