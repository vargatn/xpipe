"""
Parameter selection in N-Dimensions
"""

import numpy as np
import itertools


def partition(lst, n):
    """
    Divides a list into N roughly equal chunks

    Examples
    --------
    Define some test list, and look at the obtained chunks with different :code:`n` values::

        >>> lst = np.arange(20)
        >>> lst
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19])
        >>> partition(lst, n=5)
        [array([0, 1, 2, 3]),
         array([4, 5, 6, 7]),
         array([ 8,  9, 10, 11]),
         array([12, 13, 14, 15]),
         array([16, 17, 18, 19])]
        >>> partition(lst, n=6)
        [array([0, 1, 2]),
         array([3, 4, 5, 6]),
         array([7, 8, 9]),
         array([10, 11, 12]),
         array([13, 14, 15, 16]),
         array([17, 18, 19])]

    As we can see, even when :code:`n` is not a divisor of :code:`len(lst)`, it returns
    roughly balanced chunks

    Parameters
    ----------
    lst : list
        list to split up
    n : int
        chunks to make

    Returns
    -------
    list of lists
        list of chunks
    """
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))]
            for i in range(n) ]


def safedivide(x, y, eps=1e-8):
    """
    Calculates x / y for arrays, setting result to zero if x ~ 0 OR y ~ 0

    Defined such that

        :math:`5 / 5 \\rightarrow 1`

    but
    
        :math:`5 / 0 \\rightarrow 0` and :math:`0 / 5 \\rightarrow 0` and :math:`0 / 0 \\rightarrow 0`


    Parameters
    ----------
    x : np.array
    y : np.array
    eps : float
        threshold for setting element to zero

    Returns
    -------
    np.array
        x / y where (xabs > eps) and (yabs > eps), 0 elsewhere
    """
    xabs = np.abs(x)
    yabs = np.abs(y)
    gind = np.where((xabs > eps) * (yabs > eps))

    res = np.zeros(shape=xabs.shape)
    res[gind] = x[gind] / y[gind]
    return res


def selector(pps, limits):
    """
    Applies selection to array based on the passed parameter limits

    Selection is defined like :math:`x\\in[x_0;\\;x_1)`

    Examples
    --------

    First make some mock data::

        np.random.seed(seed = 5)
        nrows = 100000
        ids = np.arange(nrows, dtype=int)
        ra = np.random.uniform(low=0., high=360., size=nrows)
        dec = np.random.uniform(low=-60., high=10., size=nrows)
        z = np.random.uniform(low=0.1, high=1.0, size=nrows)
        lamb = np.random.uniform(low=20, high=40, size=nrows)

        data = np.vstack((ra, dec, z, lamb)).T

    then define binning edges::

        ra_edges = (0., 100., 300.)
        dec_edges = (-60., -30., 10.)
        z_edges = (0.2, 0.4, 0.8)
        lamb_edges = (20., 30., 45., 60.)

        edges = (ra_edges, dec_edges, z_edges, lamb_edges)

    This selection divides the data into 24 bins in 4-D space, which we
    can obtain perform as::

        sinds, bounds, plpairs = sl.selector(data, edges)

    Let's inspect the output, :code:`sinds` contains the boolean indexing arrays to
    select rows by, while :code:`bounds` shows the corresponding parameter boundaries,
    for example::

        >>> bounds[13]
        ((0, (100.0, 300.0)), (1, (-60.0, -30.0)), (2, (0.2, 0.4)), (3, (30.0, 45.0)))

    where the first number indicates the column of the data file it corresponds to, and the
    tuple contains the parameter boundaries for that column. Indeed we find::

        >>> data[sinds[13], :].min(axis=0)
        array([ 100.03881951,  -59.96803764,    0.20010631,   30.02361767])
        >>> data[sinds[13], :].max(axis=0)
        array([ 299.68649402,  -30.00117535,    0.39994843,   44.99060231])

    which are the boundaries we were expecting

    Parameters
    ----------
    pps : np.array N-D
        numpy array with N rows, containing the parameters to split the sample by
    limits : list of lists
        list of parameter limits, each element contains limits for a columna of pps

    Returns
    -------
    list, list, list
        list of boolean indices (one for each selection),
        list of param limits corresponding to each selection
        parameter limits expanded from sequence to list of tuples
    """

    parinds = np.arange(pps.shape[1], dtype=int)
    plpairs = []
    for pp, limit in zip(parinds, limits):
        plpair = []
        for i, lim in enumerate(limit[:-1]):
            plpair.append((pp, (limit[i], limit[i + 1]), i))
        plpairs.append(plpair)

    sinds = []
    bounds = []
    for pval in itertools.product(*plpairs):
        sval = np.ones(len(pps), dtype=bool)
        bounds.append(pval)
        for (pind, limit, tmp) in pval:
            sval *= (limit[0] <= pps[:, pind]) * (pps[:, pind] < limit[1])
        sinds.append(sval)

    sinds = np.array(sinds)
    return sinds, bounds, plpairs


def matchdd(pars, refpars, win=None, wref=None, bins=30):
    """
    Matches two D-dimensional distributions by reweighting individual objects

    New weights are assigned by comparing the normalized, weighted histograms
    of the two datasets

    Parameters
    ----------
    pars : np.ndarray
        data table to be reweighted, shape (N1, D)
    refpars : np.ndarray
        reference data table, shape (N2, D)
    win : np.array
        weights for the data table
    wref : np.array
        weights for the reference table
    bins : int or tuple
        number of bins, or tuple of bin edges

    Returns
    -------
    np.array
        new weights for the input catalog

    Notes
    ------
    *TODO: TO BE TESTED*

    """
    if win is None:
        win = np.ones(len(pars))
    if wref is None:
        wref = np.ones(len(refpars))

    refcounts, bin_edges = np.histogramdd(refpars, bins=bins, weights=wref, normed=True)
    counts = np.histogramdd(pars, bins=bin_edges, weights=win, normed=True)[0]
    wratio = safedivide(refcounts, counts)

    _digits = []
    for icol in np.arange(pars.shape[1]):
        _digits.append(np.digitize(pars[:, icol], bins=bin_edges[icol]))
    digits = np.vstack(_digits).T

    ww = np.zeros(len(pars))
    for i, dig in enumerate(digits):
        # checking that point is within bounds for each digitize...
        checkval = True
        for icol in np.arange(pars.shape[1]):
            if dig[icol] == 0 or dig[icol] == len(bin_edges[icol]):
                checkval *= False

        if checkval:
            val = tuple(np.array(dig) - 1)
            ww[i] = wratio[val] * win[i]

    return ww