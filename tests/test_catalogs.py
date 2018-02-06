"""
PyTest file for tools.catalogs
"""

import pytest
import numpy as np
import xpipe.tools.catalogs as catalogs


@pytest.fixture
def mock_data():
    np.random.seed(seed = 5)
    nrows = 100
    ids = np.arange(nrows, dtype=int)
    ra = np.random.uniform(low=0., high=360., size=nrows)
    dec = np.random.uniform(low=-60., high=10., size=nrows)
    z = np.random.uniform(low=0.1, high=1.0, size=nrows)

    mags = np.random.normal(loc=-22., scale=0.5, size=(nrows, 4))

    raw_data = np.zeros(nrows, dtype=[("ID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("MAG", "f8", 4), ("Z", "f8")])
    raw_data["ID"] = ids
    raw_data["RA"] = ra
    raw_data["DEC"] = dec
    raw_data["MAG"] = mags
    raw_data["Z"] = z

    answer_data = np.hstack((ids[:, np.newaxis], ra[:, np.newaxis], dec[:, np.newaxis], mags, z[:, np.newaxis]))

    return raw_data, answer_data


class TestToPandas(object):

    def test_content(self, mock_data):
        """check if contents are correctly copied over"""
        raw_data, answer_data = mock_data
        data = catalogs.to_pandas(raw_data)

        np.testing.assert_allclose(data.values, answer_data)

    def test_dtype(self, mock_data):
        """check if columns are appropriate"""
        raw_data, answer_data = mock_data
        data = catalogs.to_pandas(raw_data)

        columns = np.array(data.columns)
        answer_columns = np.array(["ID", "RA", "DEC", "MAG_0", "MAG_1", "MAG_2", "MAG_3", "Z"])

        for s1, s2 in zip(columns, answer_columns):
            assert s1 == s2, "dtypes not identical: " + s1 + " " + s2





