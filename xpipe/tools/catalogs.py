"""
Handles Fits -> Pandas transformations
"""

import numpy as np
import pandas as pd


def to_pandas(recarr):
    """
    Converts potentially nested record array (such as a FITS Table) into Pandas DataFrame

    FITS tables sometimes have multidimensional columns, which are not supported for DataFrames
    Pandas DataFrames however provide many nice features, such as SQL speed database matchings.

    The approach is to flatten out multidimensional column [[COL]] into [COL_1, COL_2, ..., COL_N]

    Examples
    --------

    Just pass the loaded FITS table::

        import fitsio as fio
        import xpipe.io.catalogs as catalogs

        raw_data = fio.read("data.fits")
        data = catalogs.to_pandas(raw_data)


    Parameters
    ----------
    recarr : numpy.array
        array to be converted to DataFrame

    Returns
    -------
    pandas.DataFrame
        array as DataFrame

    """

    newarr = flat_copy(recarr)
    res = pd.DataFrame.from_records(newarr.byteswap().newbyteorder(), columns=newarr.dtype.names)
    return res


def flat_type(recarr):
    """
    Assigns the dtypes to the flattened array

    Parameters
    ----------
    recarr : numpy.array
        array to be converted to DataFrame

    Returns
    -------
    list
        dtypes of flattened array

    """

    newtype = []
    for dt in recarr.dtype.descr:
        if len(dt) == 3:
            for i in np.arange(dt[2][0]):
                newtype.append((dt[0] + '_' + str(i), dt[1]))
        else:
            newtype.append(dt)
    return newtype


def flat_copy(recarr):
    """
    Copies the record array into a new recarray which has only 1-D columns

    Parameters
    ----------
    recarr : numpy.array
        array to be converted to DataFrame

    Returns
    -------
    numpy.array
        array with 1-D columns
    """

    newtype = flat_type(recarr)
    newarr = np.zeros(len(recarr), dtype=newtype)

    oldnames = recarr.dtype.names
    j = 0
    for i, dt in enumerate(recarr.dtype.descr):
        if len(dt) == 3:
            for c in np.arange(dt[2][0]):
                #                 print newtype[j]
                newarr[newtype[j][0]] = recarr[oldnames[i]][:, c]
                j += 1

        else:
            #             print newtype[j]
            newarr[newtype[j][0]] = recarr[oldnames[i]]
            j += 1
    return newarr