"""
PyTest file for tools.selector
"""

import pytest
import numpy as np
import proclens.tools.selector as sl


class TestSafeDivide(object):
    def test_result(self):
        n = 10
        x = np.arange(n, dtype="f8")
        y = np.arange(n, dtype="f8")
        x[5] = 0
        y[9] = 0

        ans = np.ones(n)
        ans[[0, 5, 9]] = 0

        res = sl.safedivide(x, y)

        np.testing.assert_allclose(res, ans)

    def test_eps(self):

        eps = 1e-8

        n = 10
        x = np.arange(n)
        y = np.arange(n)

        x[1] = 1e5
        x[5] = 1e-9
        y[9] = 1e-10

        ans = np.ones(n)
        ans[[0, 5, 9]] = 0
        ans[1] = 1e5

        res = sl.safedivide(x, y, eps=eps)

        np.testing.assert_allclose(res, ans)


@pytest.fixture
def mock_data():
    np.random.seed(seed = 5)
    nrows = 1000
    ids = np.arange(nrows, dtype=int)
    ra = np.random.uniform(low=0., high=360., size=nrows)
    dec = np.random.uniform(low=-60., high=10., size=nrows)
    z = np.random.uniform(low=0.1, high=1.0, size=nrows)

    raw_data = np.zeros(nrows, dtype=[("ID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("Z", "f8")])
    raw_data["ID"] = ids
    raw_data["RA"] = ra
    raw_data["DEC"] = dec
    raw_data["Z"] = z

    return raw_data


class TestSelector(object):
    def test_1d(self, mock_data):
        raw_data = mock_data

        data1 = raw_data[["Z"]].astype("f8")[:, np.newaxis]
        limits1 = [(0.2, 0.4, 0.6, 0.8)]

        ans_bounds = [((0, (0.2, 0.4), 0,),), ((0, (0.4, 0.6), 1),), ((0, (0.6, 0.8), 2),)]
        ans_plpairs = [[(0, (0.2, 0.4), 0), (0, (0.4, 0.6), 1), (0, (0.6, 0.8), 2)]]
        ans_sinds = [
            (0.2 <= raw_data["Z"]) * (raw_data["Z"] < 0.4),
            (0.4 <= raw_data["Z"]) * (raw_data["Z"] < 0.6),
            (0.6 <= raw_data["Z"]) * (raw_data["Z"] < 0.8),
        ]
        sinds, bounds, plpairs = sl.selector(data1, limits1)

        np.testing.assert_allclose(sinds, ans_sinds)

        assert plpairs == ans_plpairs
        assert bounds == ans_bounds

    def test_2d(self, mock_data):
        raw_data = mock_data
        limits2 = [(0., 200, 300, 360.), (-60., -30., 10.)]
        data2 = np.vstack((raw_data["RA"], raw_data["DEC"])).T

        ans_plpairs = [[(0, (0.0, 200), 0), (0, (200, 300), 1), (0, (300, 360.0), 2)],
                       [(1, (-60.0, -30.0), 0), (1, (-30.0, 10.0), 1)]]

        ans_bounds = [
            ((0, (0.0, 200), 0), (1, (-60.0, -30.0), 0)),
            ((0, (0.0, 200), 0), (1, (-30.0, 10.0), 1)),
            ((0, (200, 300), 1), (1, (-60.0, -30.0), 0)),
            ((0, (200, 300), 1), (1, (-30.0, 10.0), 1)),
            ((0, (300, 360.0), 2), (1, (-60.0, -30.0), 0)),
            ((0, (300, 360.0), 2), (1, (-30.0, 10.0), 1))
        ]
        ans_sinds = np.array([
            (0. <= raw_data["RA"]) * (raw_data["RA"] < 200.) * (-60. <= raw_data["DEC"]) * (raw_data["DEC"] < -30.),
            (0. <= raw_data["RA"]) * (raw_data["RA"] < 200.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.),
            (200. <= raw_data["RA"]) * (raw_data["RA"] < 300.) * (-60. <= raw_data["DEC"]) * (raw_data["DEC"] < -30.),
            (200. <= raw_data["RA"]) * (raw_data["RA"] < 300.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.),
            (300. <= raw_data["RA"]) * (raw_data["RA"] < 360.) * (-60. <= raw_data["DEC"]) * (raw_data["DEC"] < -30.),
            (300. <= raw_data["RA"]) * (raw_data["RA"] < 360.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.),
        ])

        sinds, bounds, plpairs = sl.selector(data2, limits2)

        np.testing.assert_allclose(sinds, ans_sinds)
        assert plpairs == ans_plpairs
        assert bounds == ans_bounds

    def test_3d(self, mock_data):
        raw_data = mock_data
        data3 = np.vstack((raw_data["RA"], raw_data["DEC"], raw_data["Z"])).T
        limits3 = [(0., 200, 360.), (-60., -30., 10.), (0.2, 0.4, 0.8)]
        sinds, bounds, plpairs = sl.selector(data3, limits3)

        ans_plpairs = [[(0, (0.0, 200), 0), (0, (200, 360.0), 1)],
                       [(1, (-60.0, -30.0), 0), (1, (-30.0, 10.0), 1)],
                       [(2, (0.2, 0.4), 0), (2, (0.4, 0.8), 1)]]
        ans_bounds = [
             ((0, (0.0, 200), 0), (1, (-60.0, -30.0), 0), (2, (0.2, 0.4), 0)),
             ((0, (0.0, 200), 0), (1, (-60.0, -30.0), 0), (2, (0.4, 0.8), 1)),
             ((0, (0.0, 200), 0), (1, (-30.0, 10.0), 1), (2, (0.2, 0.4), 0)),
             ((0, (0.0, 200), 0), (1, (-30.0, 10.0), 1), (2, (0.4, 0.8), 1)),
             ((0, (200, 360.0), 1), (1, (-60.0, -30.0), 0), (2, (0.2, 0.4), 0)),
             ((0, (200, 360.0), 1), (1, (-60.0, -30.0), 0), (2, (0.4, 0.8), 1)),
             ((0, (200, 360.0), 1), (1, (-30.0, 10.0), 1), (2, (0.2, 0.4), 0)),
             ((0, (200, 360.0), 1), (1, (-30.0, 10.0), 1), (2, (0.4, 0.8), 1))
        ]

        ans_sinds = np.array([
            (0. <= raw_data["RA"]) * (raw_data["RA"] < 200.) * (-60. <= raw_data["DEC"]) * (raw_data["DEC"] < -30.) * (
            0.2 <= raw_data["Z"]) * (raw_data["Z"] < 0.4),
            (0. <= raw_data["RA"]) * (raw_data["RA"] < 200.) * (-60. <= raw_data["DEC"]) * (raw_data["DEC"] < -30.) * (
            0.4 <= raw_data["Z"]) * (raw_data["Z"] < 0.8),
            (0. <= raw_data["RA"]) * (raw_data["RA"] < 200.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.) * (
            0.2 <= raw_data["Z"]) * (raw_data["Z"] < 0.4),
            (0. <= raw_data["RA"]) * (raw_data["RA"] < 200.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.) * (
            0.4 <= raw_data["Z"]) * (raw_data["Z"] < 0.8),
            (200. <= raw_data["RA"]) * (raw_data["RA"] < 360.) * (-60. <= raw_data["DEC"]) * (
            raw_data["DEC"] < -30.) * (0.2 <= raw_data["Z"]) * (raw_data["Z"] < 0.4),
            (200. <= raw_data["RA"]) * (raw_data["RA"] < 360.) * (-60. <= raw_data["DEC"]) * (
            raw_data["DEC"] < -30.) * (0.4 <= raw_data["Z"]) * (raw_data["Z"] < 0.8),
            (200. <= raw_data["RA"]) * (raw_data["RA"] < 360.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.) * (
            0.2 <= raw_data["Z"]) * (raw_data["Z"] < 0.4),
            (200. <= raw_data["RA"]) * (raw_data["RA"] < 360.) * (-30. <= raw_data["DEC"]) * (raw_data["DEC"] < 10.) * (
            0.4 <= raw_data["Z"]) * (raw_data["Z"] < 0.8),
        ])

        np.testing.assert_allclose(sinds, ans_sinds)
        assert plpairs == ans_plpairs
        assert bounds == ans_bounds


def test_partition():

    lst = np.arange(200)

    for n in np.arange(1, 200):
        chunks = sl.partition(lst, n=n)

        assert len(chunks) == n

        lens = [len(chunk) for chunk in chunks]
        assert np.sum(lens) == len(lst)




