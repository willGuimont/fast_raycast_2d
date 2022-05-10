import numbers
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from shapely import geometry


def line_intersect(a, b, c, d) -> (float, float):
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


def naive_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> np.ndarray:
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


def naive_four_way_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> [np.ndarray]:
    hits = []
    for delta_angle in [i * np.pi / 2 for i in range(4)]:
        hits.append(naive_raycast(m, pos, angle + delta_angle, ray_length))
    return hits


def fast_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> np.ndarray:
    """
    Computes all the hits of a ray cast from `pos` with orientation `angle` for `ray_length` units with the map `m`

    The algorithm is based on the fact that for a ray to intersect with a line, at least one of its point should be
    in front of `pos`. Also, one of those point must be at the left of the `pos`, and the other on the right.

    This allows us to filter a lot of points and only compute the intersection with lines we know we could intersect
    with our ray.
    :param m: map
    :param pos: robot's position
    :param angle: robot's orientation
    :param ray_length: length of the ray
    :return: list of all hits
    """
    # direction vectors
    forward = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([[np.cos(angle + np.pi / 2)], [np.sin(angle + np.pi / 2)]])

    # compute which points are in front of `pos`
    is_forward = np.zeros(m.shape[1])
    is_forward[np.dot((m - pos[..., None]).T, forward) > 0] = 1.0
    # convolution allows to get all lines with at least one point in front of `pos`
    # the convolution returns True if either the point to the left or right is in front
    is_forward = np.convolve(is_forward, np.array([1.0, 1.0, 1.0]), 'same') > 0
    forward_pts = np.nonzero(is_forward)[0]

    # compute points on the left of pos
    is_left = np.dot((m[:, forward_pts].T - pos), left)[:, 0] > 0

    # compute starting point of lines that cross
    # np.diff allows to find sequential points that are on different sides of `pos`
    # since left = 1 and right = 0, a non-zero diff means we passed from left to right
    start_pts_idx = np.nonzero(np.diff(is_left))[0]

    # find the hits between the ray and the filtered lines
    hits = []
    ray_end = pos + forward * ray_length
    for pt in forward_pts[start_pts_idx]:
        hit = line_intersect(m[:, pt], m[:, pt + 1], pos, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def fast_four_helper(in_front, is_at_left, m, pos, ray_end) -> np.ndarray:
    is_in_front = np.convolve(in_front, np.array([1.0, 1.0, 1.0]), 'same') > 0
    front_pts = np.nonzero(is_in_front)[0]
    start_pts_idx = np.nonzero(np.diff(is_at_left[front_pts]))[0]
    hits = []
    for pt in front_pts[start_pts_idx]:
        hit = line_intersect(m[:, pt], m[:, pt + 1], pos, ray_end)
        if hit is not None:
            hits.append(hit)
    return np.array(hits).T


def fast_four_way_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> [np.ndarray]:
    """
    Uses the same technique as `fast_raycast` but reuse computation for each cast
    :param m: map
    :param pos: robot's position
    :param angle: robot's orientation
    :param ray_length: length of the ray
    :return: list of all hits per direction [forward, left, backward, right]
    """
    forward = np.array([np.cos(angle), np.sin(angle)])
    left = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

    # forward/backward
    is_forward = np.zeros(m.shape[1])
    is_forward[np.dot((m - pos[..., None]).T, forward) > 0] = 1.0
    is_backward = 1 - is_forward

    # left/right
    is_left = np.zeros(m.shape[1])
    is_left[np.dot((m - pos[..., None]).T, left) > 0] = 1.0
    is_right = 1 - is_left

    # throw 4 rays, reuse position matrices to save compute
    hits = [fast_four_helper(is_forward, is_left, m, pos, pos + forward * ray_length),  # forward
            fast_four_helper(is_left, is_backward, m, pos, pos + left * ray_length),  # left
            fast_four_helper(is_backward, is_right, m, pos, pos - forward * ray_length),  # backward
            fast_four_helper(is_right, is_forward, m, pos, pos - left * ray_length)]  # right
    return hits


def shapely_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> np.ndarray:
    ray_direction = np.array([np.cos(angle), np.sin(angle)])
    ray_start, ray_end = pos, pos + ray_length * ray_direction
    ray = geometry.LineString([ray_start, ray_end])
    polygon = geometry.LineString(m.T)
    hits = polygon.intersection(ray)
    if isinstance(hits, geometry.Point):
        return np.array(hits.xy)
    if isinstance(hits, geometry.LineString):
        return np.empty((2, 0))
    else:
        return np.array([x.xy for x in hits.geoms]).T.squeeze()


def shapely_four_way_raycast(m: np.ndarray, pos: np.ndarray, angle: float, ray_length: float) -> [np.ndarray]:
    hits = []
    for delta_angle in [i * np.pi / 2 for i in range(4)]:
        hits.append(shapely_raycast(m, pos, angle + delta_angle, ray_length))
    return hits


def _plot_map(m: np.ndarray, pos: np.ndarray, angle: Union[float, List[float]], ray_length: float):
    if isinstance(angle, numbers.Number):
        angle = [angle]

    for a in angle:
        ray_direction = np.array([np.cos(a), np.sin(a)])
        ray_start, ray_end = pos, pos + ray_length * ray_direction
        ray = np.stack([ray_start, ray_end]).T
        plt.plot(ray[0, :], ray[1, :], marker='', c='r', label='ray')

    plt.plot(m[0, :], m[1, :], 'b-')
    plt.plot(pos[0], pos[1], marker='o', c='r')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')


def _assert_hit_validity(ground_truth: np.ndarray, values: [np.ndarray]):
    ground_truth.sort()
    for v in values:
        v.sort()
        assert np.all(np.isclose(ground_truth, v)), "algorithms differ"


def _assert_four_way_hit_validity(ground_truth: [np.ndarray], values: [[np.ndarray]]):
    for gt, *vals in zip(ground_truth, *values):
        gt.sort()
        for v in vals:
            v.sort()
            assert np.all(np.isclose(gt, v)), "algorithms differ"


if __name__ == '__main__':
    import timeit
    import tqdm

    mat_data = scio.loadmat('data/map.mat')
    map_polygon = mat_data['Carte']

    pos = np.array([11, 2])
    angle = np.pi / 2
    ray_length = 8

    # Simple case
    naive_hits = naive_raycast(map_polygon, pos, angle, ray_length)
    fast_hits = fast_raycast(map_polygon, pos, angle, ray_length)
    shapely_hits = shapely_raycast(map_polygon, pos, angle, ray_length)

    _plot_map(map_polygon, pos, angle, ray_length)
    plt.scatter(naive_hits[0, :], naive_hits[1, :], marker='x', c='r', label='Naive')
    plt.scatter(fast_hits[0, :], fast_hits[1, :], marker='x', c='b', label='Fast')
    plt.scatter(shapely_hits[0, :], shapely_hits[1, :], marker='x', c='g', label='Shapely')

    plt.legend()
    plt.show()

    _assert_hit_validity(naive_hits, [fast_hits, shapely_hits])

    # Four way
    naive_four_hits = naive_four_way_raycast(map_polygon, pos, angle, ray_length)
    fast_four_hits = fast_four_way_raycast(map_polygon, pos, angle, ray_length)
    shapely_four_hits = shapely_four_way_raycast(map_polygon, pos, angle, ray_length)

    _assert_four_way_hit_validity(naive_four_hits, [fast_four_hits, shapely_four_hits])

    _plot_map(map_polygon, pos, [angle + i * np.pi / 2 for i in range(4)], ray_length)
    for x in naive_four_hits:
        if x.shape[0] != 0:
            plt.scatter(x[0, :], x[1, :], marker='x', c='r')
    for x in fast_four_hits:
        if x.shape[0] != 0:
            plt.scatter(x[0, :], x[1, :], marker='x', c='b')
    for x in shapely_four_hits:
        if x.shape[0] != 0:
            plt.scatter(x[0, :], x[1, :], marker='x', c='g')

    plt.show()

    # Simple speed test
    num_runs = 10_000
    print(f'Running speed test with {num_runs} trials')
    time_naive = timeit.timeit(lambda: naive_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
    time_fast = timeit.timeit(lambda: fast_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
    time_shapely = timeit.timeit(lambda: shapely_raycast(map_polygon, pos, angle, ray_length), number=num_runs)

    print()
    print(f'Naive time for {num_runs} runs: {time_naive}s')
    print(f'Fast time for {num_runs} runs: {time_fast}s')
    print(f'Shapely time for {num_runs} runs: {time_shapely}s')
    print(f'Naive / Fast = {time_naive / time_fast}')
    print(f'Shapely / Fast = {time_shapely / time_fast}')

    # Algorithm validation
    print()
    print('Checking for algorithm error')
    for i in tqdm.tqdm(range(50_000)):
        pos = np.array([np.random.uniform(-2, 20), np.random.uniform(-2, 9)])
        angle = np.random.uniform(0, 2 * np.pi)

        naive_hits = naive_raycast(map_polygon, pos, angle, ray_length)
        fast_hits = fast_raycast(map_polygon, pos, angle, ray_length)
        shapely_hits = shapely_raycast(map_polygon, pos, angle, ray_length)

        _assert_hit_validity(naive_hits, [fast_hits, shapely_hits])

        naive_four_hits = naive_four_way_raycast(map_polygon, pos, angle, ray_length)
        fast_four_hits = fast_four_way_raycast(map_polygon, pos, angle, ray_length)
        shapely_four_hits = shapely_four_way_raycast(map_polygon, pos, angle, ray_length)

        # shapely returns incorrect results in some cases
        _assert_four_way_hit_validity(naive_four_hits, [fast_four_hits, shapely_four_hits])

    # Full speed test
    print()
    print('Comparing speed for random positions and orientations')
    times_naive = []
    times_fast = []
    times_shapely = []
    times_four_naive = []
    times_four_fast = []
    times_four_shapely = []
    for i in tqdm.tqdm(range(10_000)):
        pos = np.array([np.random.uniform(0, 18), np.random.uniform(0, 7)])
        angle = np.random.uniform(0, 2 * np.pi)

        num_runs = 10
        time_naive = timeit.timeit(lambda: naive_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
        time_fast = timeit.timeit(lambda: fast_raycast(map_polygon, pos, angle, ray_length), number=num_runs)
        time_shapely = timeit.timeit(lambda: shapely_raycast(map_polygon, pos, angle, ray_length), number=num_runs)

        time_four_naive = timeit.timeit(lambda: naive_four_way_raycast(map_polygon, pos, angle, ray_length),
                                        number=num_runs)
        time_four_fast = timeit.timeit(lambda: fast_four_way_raycast(map_polygon, pos, angle, ray_length),
                                       number=num_runs)
        time_four_shapely = timeit.timeit(lambda: shapely_four_way_raycast(map_polygon, pos, angle, ray_length),
                                          number=num_runs)

        times_naive.append(time_naive)
        times_fast.append(time_fast)
        times_shapely.append(time_shapely)

        times_four_naive.append(time_four_naive)
        times_four_fast.append(time_four_fast)
        times_four_shapely.append(time_four_shapely)

    print(f'Single: \tNaive / Fast = {np.sum(times_naive) / np.sum(times_fast)}')
    print(f'Single: \tShapely / Fast = {np.sum(times_shapely) / np.sum(times_fast)}')
    print(f'Four way: \tNaive / Fast = {np.sum(times_four_naive) / np.sum(times_four_fast)}')
    print(f'Four way: \tShapely / Fast = {np.sum(times_four_shapely) / np.sum(times_four_fast)}')
