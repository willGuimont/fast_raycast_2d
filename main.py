import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def plot_map(m: np.ndarray):
    plt.plot(m[0, :], m[1, :], 'b-')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')


def line_intersect(a, b, c, d):
    ax1, ay1 = a
    ax2, ay2 = b
    bx1, by1 = c
    bx2, by2 = d
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        u_a = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        u_b = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None
    if not (0 <= u_a <= 1 and 0 <= u_b <= 1):
        return None
    x = ax1 + u_a * (ax2 - ax1)
    y = ay1 + u_a * (ay2 - ay1)

    return x, y


def naive_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float):
    ray_direction = np.array([np.cos(angle), np.sin(angle)])
    ray_start, ray_end = pos, pos + ray_length * ray_direction
    hits = []
    num_pts = m.shape[1]
    for i in range(num_pts - 1):
        a, b = m[:, i], m[:, i + 1]
        hit = line_intersect(a, b, ray_start, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def fast_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float):
    forward = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([[np.cos(angle + np.pi / 2)], [np.sin(angle + np.pi / 2)]])

    # front
    in_front = np.zeros(m.shape[1])
    in_front[np.dot((m - pos[..., None]).T, forward) > 0] = 1.0
    is_in_front = np.convolve(in_front, np.array([1.0, 1.0, 1.0]), 'same') > 0
    front_pts = np.nonzero(is_in_front)[0]

    # left
    is_at_left = np.dot((m[:, front_pts].T - pos), left)[:, 0] > 0
    start_pts_idx = np.nonzero(np.diff(is_at_left))[0]

    hits = []
    ray_end = pos + forward * ray_length
    for pt in front_pts[start_pts_idx]:
        hit = line_intersect(m[:, pt], m[:, pt + 1], pos, ray_end)
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
    ray_length = 200

    naive_hits = naive_raycast(map_polygon, pos, angle, ray_length)
    fast_hits = fast_raycast(map_polygon, pos, angle, ray_length)

    print(f'Are outputs the same for simple case: {np.all(naive_hits == fast_hits)}')

    plot_map(map_polygon)
    plt.scatter(naive_hits[0, :], naive_hits[1, :], marker='x', c='r', label='Naive')
    plt.scatter(fast_hits[0, :], fast_hits[1, :], marker='x', c='b', label='Fast')

    plt.legend()
    plt.show()

    num_runs = 50_000
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
        assert np.all(np.isclose(naive_hits, fast_hits)), "algorithms differ"

    print()
    print('Comparing speed for random positions and orientations')
    times_naive = []
    times_fast = []
    for i in tqdm.tqdm(range(10_000)):
        pos = np.array([np.random.uniform(0, 18), np.random.uniform(0, 7)])
        angle = np.random.uniform(0, 2 * np.pi)

        num_runs = 10
        time_naive = timeit.timeit(lambda: naive_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
        time_fast = timeit.timeit(lambda: fast_raycast(map_polygon, pos, angle, ray_length), number=num_runs)

        times_naive.append(time_naive)
        times_fast.append(time_fast)
    print(f'Naive / Fast = {np.sum(times_naive) / np.sum(times_fast)}')
