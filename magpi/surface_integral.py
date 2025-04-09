import math

from .prelude import *
from .integrate import integrate, gauss
from .calc import value_and_jacfwd


def integrate_surface(f, parametrization, u, v, *args, method=gauss(5), **kwargs):
    def integrand(x):
        J = jacfwd(parametrization)(x)
        msg = "Please provide a surface parametrization from [0,1]² -> R³"
        assert J.shape == (3, 2), msg
        ru = J[:, 0]
        rv = J[:, 1]
        z = norm(cross(ru, rv))
        r = parametrization(x)
        return tree_map(lambda y: y * z, f(r, *args, **kwargs))

    return integrate(integrand, [u, v], method=method)


def integrate_surface_elements(f, parametrization, u, v, *args, method=gauss(5), **kwargs):
    def _integrate(u, v):
        c = center(parametrization, u, v)
        return integrate_surface(f, parametrization, u, v, c, *args, method=method, **kwargs)

    assert len(u.shape) == len(v.shape) == 1
    assert u.shape[0] >= 2
    assert v.shape[0] >= 2
    _u = jnp.stack([u[:-1], u[1:]], axis=-1)
    _v = jnp.stack([v[:-1], v[1:]], axis=-1)
    return vmap(vmap(_integrate, (None, 0)), (0, None))(_u, _v)


def center(parametrization, u, v):
    u_bar = (u[..., 1] + u[..., 0]) / 2
    v_bar = (v[..., 1] + v[..., 0]) / 2
    return parametrization(jnp.stack([u_bar, v_bar], axis=-1))


def midpoints(parametrization, u, v):
    def _center(u, v):
        return center(parametrization, u, v)

    _u = jnp.stack([u[:-1], u[1:]], axis=-1)
    _v = jnp.stack([v[:-1], v[1:]], axis=-1)
    return vmap(vmap(_center, (None, 0)), (0, None))(_u, _v)

    
def _source_tensor(x, parametrization, u, v, *, method=gauss(5), order=2, expsum_coefs=None):
    assert order >= 0

    def _integrand(y, c, x):
        assert y.shape == (3,)
        assert x.shape == (3,)
        assert c.shape == (3,)
        if expsum_coefs is None:
            n = 1 / norm(x - y)
        else:
            omega, alpha = expsum_coefs

            n = expsum(norm(x - y), omega, alpha)

        c1 = asarray([1.0])
        d = y - c
        t = d
        T = [c1]
        for i in range(0, order):
            T.append(t.ravel())
            t = jnp.tensordot(t, d, 0)

        c = jnp.concatenate(T, axis=-1)
        return c * n

    Z = integrate_surface_elements(_integrand, parametrization, u, v, x, method=method)
    return Z.reshape(-1)


@partial(jit, static_argnames=("parametrization", "method", "order", "compute_jacfwd"))
def source_tensor(x, parametrization, u, v, *, method=gauss(5), order=2, compute_jacfwd=True, expsum_coefs=None):
    if compute_jacfwd:
        return value_and_jacfwd(_source_tensor)(x, parametrization, u, v, method=method, order=order, expsum_coefs=expsum_coefs)
    else:
        return _source_tensor(x, parametrization, u, v, method=method, order=order, expsum_coefs=expsum_coefs)
    

@partial(jit, static_argnames=("f", "parametrization", "order"))
def charge_tensor(f, parametrization, u, v, *args, order=2, **kwargs):
    _f = lambda x: f(x, *args, **kwargs)
    assert order >= 0

    def charge(c):
        v0 = asarray(_f(c))
        v = v0
        df = _f
        F = [v0[..., None]]
        for i in range(order):
            df = jacfwd(df)
            v = df(c) / math.factorial(i + 1)
            v = v.reshape((*v0.shape, -1))
            F.append(v)

        coefs = jnp.concatenate(F, axis=-1)
        return coefs.swapaxes(-1, 0)

    c = midpoints(parametrization, u, v)
    F = jnp.apply_along_axis(charge, -1, c)
    base_dim = len(c.shape[:-1])
    return F.reshape(-1, *F.shape[base_dim + 1:])


def single_layer_potential(source_tensor, charge_tensor):
    return 1 / (4 * pi) * jnp.tensordot(charge_tensor, source_tensor, ((0,), (0,)))


def curl_single_layer_potential(source_tensor_derivative, charge_tensor):
    curl_s = -jnp.cross(charge_tensor, source_tensor_derivative)
    return 1 / (4 * pi) * jnp.sum(curl_s, 0)


def scalar_potential_charge(adf, mag, phi1, normalized=False):
    if normalized:
        def charge(x, params_mag=(), params_phi1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)  # it is still important to divide by norm(n) for AD
            return mag(x, *params_mag) @ n + phi1(x, *params_phi1)
        
        return charge
    else:
        def charge(x, params_mag=(), params_phi1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)
            return mag(x, *params_mag) @ n - jacfwd(phi1)(x, *params_phi1) @ n
        
        return charge
        
    
def vector_potential_charge(adf, mag, A1, normalized=False):
    if normalized:
        def charge(x, params_mag=(), params_A1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)
            return cross(mag(x, *params_mag), n) + A1(x, *params_A1)
        
        return charge
    else:
        def charge(x, params_mag=(), params_A1=()):
            n = -jacfwd(adf)(x)
            n = n / norm(n)
            return cross(mag(x, *params_mag), n) - jacfwd(A1)(x, *params_A1) @ n
        
        return charge
    

def tri_parametrization(x, v1, v2, v3):
    a, b = x[0], x[1]
    l1, l2, l3 = 1 - sqrt(a), (1 - b) * sqrt(a), b * sqrt(a)
    return l1 * v1 + l2 * v2 + l3 * v3


def source_tensor_for_mesh(x, mesh, *, order=2, method=gauss(5), compute_jacfwd=True, expsum_coefs=None):
    u = v = jnp.linspace(0, 1, 2)
    def source_for_element(i):
        v1, v2, v3 = mesh.nodes[i]
        p = lambda x: tri_parametrization(x, v1, v2, v3)
        z = source_tensor(x, p, u, v, method=method, order=order, compute_jacfwd=compute_jacfwd, expsum_coefs=expsum_coefs)
        return z

    if compute_jacfwd:
        z, dz = vmap(source_for_element)(mesh.elements)
        z = z.reshape(-1)
        dz = dz.reshape(-1, 3)
        return z, dz
    else:
        z = vmap(source_for_element)(mesh.elements)
        z = z.reshape(-1)
        return z
    

def charge_tensor_for_mesh(f, mesh, *args, order=2, **kwargs):
    u = v = jnp.linspace(0, 1, 2)
    def inner(i):
        v1, v2, v3 = mesh.nodes[i]
        p = lambda x: tri_parametrization(x, v1, v2, v3)
        s = charge_tensor(f, p, u, v, *args, order=order, **kwargs)
        return s
    s = vmap(inner)(mesh.elements)
    return s.reshape(-1)


OMEGA = [
    0.0000424477324713479482292261821464078933,
    0.0000477841483785349179990204388667656213,
    0.0000615134022358309026125797473651818370,
    0.0000872486060192464157340710048898872742,
    0.0001277116990755026831409746617546074887,
    0.0001868285553651566400008921176511753059,
    0.0002709373745971005086841693635041697363,
    0.0003890695113880716645428964637698845963,
    0.0005534592955142687855098273814590664754,
    0.0007804003798261143429438899750578972903,
    0.0010914167878501818044198600908718799207,
    0.0015147769293589993331792037585016574841,
    0.0020874197762817817225016830918679633378,
    0.0028573894938656410826173802407559887406,
    0.0038869001630011921059395260755892076432,
    0.0052561805874043536250564921923772443435,
    0.0070682827386090824882571145501564124913,
    0.0094550776998144114129837270107947766462,
    0.0125847112720908547699793047537630830135,
    0.0166708496698530177921855708472742563231,
    0.0219841156389002174476493848762070904002,
    0.0288661993253051596936930960068945495323,
    0.0377472300304736547962218638080367227872,
    0.0491671218170339489853697990950909257890,
    0.0638017766348427351226079047574746283544,
    0.0824953035727754739522008323171498034299,
    0.1062999971631043435809325925500701259807,
    0.1365274221740425074887465700657074307856,
    0.1748190276160293898330974973753804135868,
    0.2232616728015435169350279276390125460239,
    0.2846314500771694828366430857213842386955,
    0.3630542879512230530682920237595112666895,
    0.4661677825372563676469862825602064049235,
    0.6137955140038285507656824757649616230992,
    0.8922962458720837759839142333539996343461,
]

ALPHA = [
    0.0000000003472894471897133760815249889858,
    0.0000000033807030690271800939453339773826,
    0.0000000111947686164436191330000526957295,
    0.0000000291507775254263330118128773898782,
    0.0000000701111568774001681101065962823721,
    0.0000001620746843313424328973761506492941,
    0.0000003639550840045464226913567759400810,
    0.0000007971429983891580112966924953488558,
    0.0000017069944114907889588321893455199107,
    0.0000035805601729945747400209522370495758,
    0.0000073685878454165469480486529116504543,
    0.0000148982510838220779688592655474316511,
    0.0000296303567195374358161682287434498390,
    0.0000580313947498979342402661663020491459,
    0.0001120308527028118275019168893660674591,
    0.0002133741725533846435915188324123313579,
    0.0004012542517834215356194749879529051761,
    0.0007455604090965460571931013933951598593,
    0.0013696672756435273147401960897734385370,
    0.0024892889407007671712534384918930063790,
    0.0044781708078441673927765206628115457477,
    0.0079782734579144526686412435337314441597,
    0.0140831798157676060668765322461681854804,
    0.0246411741517513898124181652773967421410,
    0.0427526132853131875306731783659142109855,
    0.0735807773481342002543808579473338937760,
    0.1256656566269579475785380176522387785099,
    0.2130410835736244252916048863566444993012,
    0.3586333527836384865684297290622239984259,
    0.5997181349484841186587430472698656558350,
    0.9967783801818366268472117186494330098867,
    1.6484481145432781049476461676661642741237,
    2.7194810226002215029338787699586532653484,
    4.5068021911540400263607164532686510938220,
    7.6854766304843686514948475352326795473346,
]

RK = 1E09


def expsum_coefs(hmax: float) -> tuple[Array, Array, float]:
    h_min = hmax / sqrt(RK)
    omega = array(OMEGA) / h_min
    alpha = array(ALPHA) / (h_min ** 2)
    return omega, alpha, float(h_min)


def expsum(x, omega, alpha):
    return jnp.sum(omega * exp(-alpha * x ** 2))
