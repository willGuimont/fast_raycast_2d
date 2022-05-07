import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def plot_map(m: np.ndarray):
    plt.plot(m[0, :], m[1, :], 'b-')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def is_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_intersect(a, b, c, d):
    Ax1, Ay1 = a
    Ax2, Ay2 = b
    Bx1, By1 = c
    Bx2, By2 = d
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)

    return x, y


def naive_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_lenght: float):
    ray_direction = np.array([np.cos(angle), np.sin(angle)])
    ray_start, ray_end = pos, pos + ray_lenght * ray_direction
    hits = []
    num_pts = m.shape[1]
    for i in range(num_pts - 1):
        a, b = m[:, i], m[:, i + 1]
        hit = line_intersect(a, b, ray_start, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def fast_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_lenght: float):
    forward = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
    ray_start, ray_end = pos, pos + ray_lenght * forward

    # front
    in_front = np.zeros(m.shape[1])
    in_front[np.dot((m - pos[..., None]).T, forward[..., None]).squeeze() > 0] = 1.0
    is_in_front = np.convolve(in_front, np.array([1, 1, 1]), 'same') > 0
    front_pts = np.nonzero(is_in_front)[0]

    # left
    is_at_left = np.dot((m[:, front_pts].T - pos[..., None].T), left[..., None]).squeeze() > 0

    start_pts_idx = np.nonzero(np.diff(is_at_left))[0]
    start_pts = front_pts[start_pts_idx]

    hits = []
    for pt in start_pts:
        hit = line_intersect(m[:, pt], m[:, pt + 1], ray_start, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


if __name__ == '__main__':
    import timeit
    import tqdm

    mat_data = scio.loadmat('data/map.mat')
    map_polygon = mat_data['Carte']

    pos = np.array([11, 3])
    angle = np.pi / 2
    ray_length = 20

    naive_hits = naive_raycast(map_polygon, pos, angle, ray_length)
    fast_hits = fast_raycast(map_polygon, pos, angle, ray_length)

    print(f'Are outputs the same for simple case: {np.all(naive_hits == fast_hits)}')

    plot_map(map_polygon)
    plt.scatter(naive_hits[0, :], naive_hits[1, :], marker='x', c='r', label='Naive')
    plt.scatter(fast_hits[0, :], fast_hits[1, :], marker='x', c='b', label='Fast')

    plt.legend()
    plt.show()

    num_runs = 10_000
    time_naive = timeit.timeit(lambda: naive_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
    time_fast = timeit.timeit(lambda: fast_raycast(map_polygon, pos, angle, ray_length), number=num_runs)

    print()
    print(f'Naive time for {num_runs} runs: {time_naive}s')
    print(f'Fast time for {num_runs} runs: {time_fast}s')
    print(f'Naive / Fast = {time_naive / time_fast}')

    print()
    print('Checking for algorithm error')
    for i in tqdm.tqdm(range(100_000)):
        pos = np.array([np.random.uniform(0, 18), np.random.uniform(0, 7)])
        angle = np.random.uniform(0, 2 * np.pi)

        naive_hits = naive_raycast(map_polygon, pos, angle, ray_length)
        fast_hits = fast_raycast(map_polygon, pos, angle, ray_length)
        assert np.all(naive_hits == fast_hits), "algorithms differ"

    print()
    print('Comparing speed for random positions and orientations')
    times_naive = []
    times_fast = []
    for i in tqdm.tqdm(range(100_000)):
        pos = np.array([np.random.uniform(0, 18), np.random.uniform(0, 7)])
        angle = np.random.uniform(0, 2 * np.pi)

        num_runs = 10
        time_naive = timeit.timeit(lambda: naive_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
        time_fast = timeit.timeit(lambda: fast_raycast(map_polygon, pos, angle, ray_length), number=num_runs)

        times_naive.append(time_naive)
        times_fast.append(time_fast)
    print(f'Naive / Fast = {np.sum(times_naive) / np.sum(times_fast)}')
