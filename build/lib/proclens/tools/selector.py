

import numpy as np
import itertools



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

    coming soon

    Parameters
    ----------
    pps : np.array N-D
        numpy array with N rows, containing the parameters to split the sample by
    limits : list of tuples
        list of parameter limits, each element contains limits for a row of pps

    Returns
    -------
    list, of bool arrays list
        list of boolean indices (one for each selection),
        list of corresponding parameter limits
    """

    parinds = np.arange(pps.shape[1], dtype=int)
    plpairs = []
    for pp, limit in zip(parinds, limits):
        plpair = []
        for i, lim in enumerate(limit[:-1]):
            plpair.append((pp, (limit[i], limit[i + 1])))
        plpairs.append(plpair)

    sinds = []
    for pval in itertools.product(*plpairs):
        sval = np.ones(len(pps), dtype=bool)
        for (pind, limit) in pval:
            sval *= (limit[0] <= pps[:, pind]) * (pps[:, pind] < limit[1])
        sinds.append(sval)

    sinds = np.array(sinds)
    return sinds, plpairs