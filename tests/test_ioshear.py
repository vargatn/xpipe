"""
PyTest file for tools.selector
"""

import pytest
import numpy as np
import xpipe.xhandle.ioshear as ioshear
import tempfile


class TestRaw(object):
    def test_data(self, tmpdir):
        dirname = str(tmpdir.mkdir("test"))
        fname = dirname + "/" + "test.dat"

        arr = np.arange(10)[:, np.newaxis]

        np.savetxt(fname, arr)

        res = ioshear.read_raw(fname)
        np.testing.assert_allclose(res, arr)


    def test_empty(self, tmpdir):
        dirname = str(tmpdir.mkdir("test"))
        fname = dirname + "/" + "test.dat"

        arr = np.array([])

        res = ioshear.read_raw(fname)

        np.testing.assert_allclose(res, arr)


class TestMultipleRaw(object):
    def test_data(self, tmpdir):
        dirname = str(tmpdir.mkdir("test"))

        nfile = 10
        ans = []

        tmp_names = []
        for i in np.arange(nfile):
            fname = dirname + "/" + str(i) + ".dat"

            arr = np.arange(i+1)[:, np.newaxis]
            np.savetxt(fname, arr)

            tmp_names.append(fname)
            ans.append(arr)

        res = ioshear.read_multiple_raw(tmp_names)

        assert len(res) == nfile
        np.testing.assert_allclose(np.vstack(res), np.vstack(ans))

    def test_empty(self, tmpdir):
        dirname = str(tmpdir.mkdir("test"))

        nfile = 10
        ans = []
        tmp_names = []
        for i in np.arange(nfile):
            fname = dirname + "/" + str(i) + ".dat"

            if i % 2 == 0:
                arr = np.arange(i+1)[:, np.newaxis]
                np.savetxt(fname, arr)
            elif i in (5, 7):
                arr = np.array([])
                np.savetxt(fname, arr)
            else:
                arr = np.array([])

            tmp_names.append(fname)
            ans.append(arr)

        res = ioshear.read_multiple_raw(tmp_names)

        assert len(res) == nfile
        for i in np.arange(nfile):
            assert len(res[i]) == len(ans[i])


def test_sheared_raw(tmpdir):
    dirname = str(tmpdir.mkdir("test"))
    shears = ioshear.sheared_tags

    arr = np.arange(10)

    snames = []
    base_name = dirname + "/test.dat"
    for i in np.arange(len(shears)):
        fname = base_name.replace(".dat", shears[i]+ ".dat")
        np.savetxt(fname, arr)
        snames.append(fname)

    res = ioshear.read_sheared_raw(base_name)
    assert len(res) == len(shears)


def test_multiple_sheared_raw(tmpdir):
    dirname = str(tmpdir.mkdir("test"))
    shears = ioshear.sheared_tags

    arr = np.arange(10)

    base_name1 = dirname + "/test1.dat"
    for i in np.arange(len(shears)):
        fname = base_name1.replace(".dat", shears[i]+ ".dat")
        np.savetxt(fname, arr)

    base_name2 = dirname + "/test2.dat"
    for i in np.arange(len(shears)):
        fname = base_name2.replace(".dat", shears[i]+ ".dat")
        np.savetxt(fname, arr)

    base_name = [base_name1, base_name2]
    res = ioshear.read_multiple_sheared_raw(base_name)
    assert len(res) == len(shears)


class TestXPatches(object):
    def test_data(self, monkeypatch):

        def mock_xread(data):
            return data, data.T[:, :, np.newaxis], None
        monkeypatch.setattr(ioshear, "xread", mock_xread)

        ans_labels = []
        raw_chunks = []
        for i in np.arange(-5, 5, 1):
            val = np.abs(i)
            arr = np.ones(shape=(val, 4)) * i
            raw_chunks.append(arr)
            ans_labels.append(np.ones(val) * i)

        ans_infos = np.vstack(raw_chunks)
        ans_datas = np.vstack(raw_chunks).T[:, :, np.newaxis]
        ans_labels = np.concatenate(ans_labels) + 5.


        infos, datas, labels = ioshear.xpatches(raw_chunks)

        np.testing.assert_allclose(infos, ans_infos)
        np.testing.assert_allclose(datas, ans_datas)
        np.testing.assert_allclose(labels, ans_labels)


