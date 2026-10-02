"""Walsh transforms and tests for level-one integer-valued Z-bent functions.

Vectors use binary integer order: index x = sum_i x_i * 2**i.
Solver routines import PuLP only when called.
"""
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / 'data'

def _integer_vector(values):
    a = np.asarray(values)
    if a.ndim != 1 or len(a) == 0 or len(a) & (len(a) - 1):
        raise ValueError('Expected a nonempty vector of power-of-two length.')
    if not np.issubdtype(a.dtype, np.integer) or not np.isin(a, [-1, 0, 1]).all():
        raise ValueError('Expected integer entries in {-1, 0, 1}.')
    return a.astype(np.int64, copy=True)

def hadamard(n):
    """Return the unnormalized Walsh matrix H_n in binary integer order."""
    if not isinstance(n, int) or n < 0:
        raise ValueError('n must be a nonnegative integer.')
    h = np.array([[1]], dtype=np.int64)
    for _ in range(n):
        h = np.kron(h, np.array([[1, 1], [1, -1]], dtype=np.int64))
    return h

def fwht(values):
    """Compute the unnormalized Walsh transform with integer arithmetic."""
    a = np.asarray(values)
    if a.ndim != 1 or len(a) == 0 or len(a) & (len(a) - 1):
        raise ValueError('Expected a nonempty vector of power-of-two length.')
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError('Expected an integer vector.')
    a = a.astype(np.int64, copy=True)
    step = 1
    while step < len(a):
        blocks = a.reshape(-1, 2 * step)
        left, right = blocks[:, :step].copy(), blocks[:, step:].copy()
        blocks[:, :step] = left + right
        blocks[:, step:] = left - right
        step *= 2
    return a

def is_level1(f, k):
    """Check both f and 2**(-k) H_(2k) f are ternary integer vectors."""
    if not isinstance(k, int) or k < 0:
        return False
    try:
        a = _integer_vector(f)
    except ValueError:
        return False
    return len(a) == 1 << (2*k) and np.isin(fwht(a), [-2**k, 0, 2**k]).all()

def is_bent_sign(a, k):
    """Check a is a sign vector of a bent Boolean function."""
    return is_level1(a, k) and np.isin(a, [-1, 1]).all() and np.all(np.abs(fwht(a)) == 2**k)

def load_data(name):
    """Load a bundled array independently of the current working directory."""
    if Path(name).name != name:
        raise ValueError('Use a bundled data filename.')
    return np.load(DATA_DIR / name, allow_pickle=False).astype(np.int64)

def _pulp():
    try:
        import pulp
    except ImportError as exc:
        raise RuntimeError('Install the optional solver requirements: pip install -r requirements-solver.txt') from exc
    if not hasattr(pulp, 'PULP_CBC_CMD'):
        raise RuntimeError('Use the pinned solver requirements: pip install -r requirements-solver.txt')
    return pulp

def split_status(f, k, H=None, timeLimit=300):
    """Return SPLIT, NONSPLIT, or UNKNOWN:<solver status>.

    CBC infeasibility is a numerical solver result, not the exact determinant
    certificate supplied by verify_seed.py. Time limits are never interpreted
    as proofs of non-splitting. A returned SPLIT witness is checked with integers.
    """
    if not is_level1(f, k):
        raise ValueError('f must be a level-one vector on 2k variables.')
    a = _integer_vector(f); N = len(a); tk = 2**k
    if H is None:
        H = hadamard(2*k)
    H = np.asarray(H)
    if H.shape != (N, N) or not np.array_equal(H, hadamard(2*k)):
        raise ValueError('H must be the Walsh matrix in binary integer order.')
    pulp = _pulp(); dual = fwht(a) // tk
    zero = np.flatnonzero(a == 0).tolist()
    if not zero:
        return 'SPLIT'
    problem = pulp.LpProblem('splitting', pulp.LpMinimize)
    b = {x: pulp.LpVariable(f'b{x}', cat='Binary') for x in zero}
    for u in range(N):
        expr = pulp.lpSum(int(H[u, x]) * (2*b[x]-1) for x in zero)
        if dual[u] != 0:
            problem += expr == 0
        else:
            v = pulp.LpVariable(f'v{u}', cat='Binary')
            problem += expr == tk*(2*v-1)
    problem += 0
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timeLimit))
    name = pulp.LpStatus[status]
    if name == 'Optimal':
        partner = np.zeros(N, dtype=np.int64)
        for x in zero:
            value = b[x].value()
            if value is None or min(abs(value), abs(value-1)) > 1e-6:
                return 'UNKNOWN:invalid witness'
            partner[x] = 2*int(round(value))-1
        if not is_bent_sign(a+partner, k) or not is_bent_sign(a-partner, k):
            return 'UNKNOWN:invalid witness'
        return 'SPLIT'
    if name == 'Infeasible':
        return 'NONSPLIT'
    return 'UNKNOWN:' + name

def splits(f, k, H=None, timeLimit=120):
    """Return a Boolean decision; raise RuntimeError for an unresolved solver run."""
    status = split_status(f, k, H=H, timeLimit=timeLimit)
    if status.startswith('UNKNOWN'):
        raise RuntimeError(status)
    return status == 'SPLIT'

def find_level1(H, k, N, obj, timeLimit=15, min_zeros=4):
    """Solve the specified integer optimization problem for a level-one vector.

    Returns a verified optimal vector, or None when optimality is not established.
    """
    if N != 1 << (2*k) or not np.array_equal(np.asarray(H), hadamard(2*k)):
        raise ValueError('The dimension and Walsh matrix must agree.')
    obj = np.asarray(obj)
    if obj.shape != (N,) or not np.issubdtype(obj.dtype, np.integer):
        raise ValueError('obj must be an integer vector of length N.')
    if not 0 <= min_zeros <= N:
        raise ValueError('min_zeros must be between 0 and N.')
    pulp = _pulp(); problem = pulp.LpProblem('level_one', pulp.LpMaximize)
    f = {x:pulp.LpVariable(f'f{x}', -1, 1, cat='Integer') for x in range(N)}
    z = {u:pulp.LpVariable(f'z{u}', -1, 1, cat='Integer') for u in range(N)}
    occupied = {x:pulp.LpVariable(f'a{x}', cat='Binary') for x in range(N)}
    for u in range(N):
        problem += pulp.lpSum(int(H[u,x])*f[x] for x in range(N)) == 2**k*z[u]
    for x in range(N):
        problem += f[x] <= occupied[x]
        problem += f[x] >= -occupied[x]
    problem += pulp.lpSum(occupied.values()) <= N-min_zeros
    problem += pulp.lpSum(int(obj[x])*f[x] for x in range(N))
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=timeLimit))
    if pulp.LpStatus[status] != 'Optimal':
        return None
    result = np.array([int(round(f[x].value())) for x in range(N)], dtype=np.int64)
    if not is_level1(result, k) or np.count_nonzero(result) > N-min_zeros:
        raise RuntimeError('The solver did not return a valid level-one vector.')
    return result
