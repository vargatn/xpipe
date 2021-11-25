"""
Plotting and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.ndimage as ndimage

import scipy.ndimage as ndimage

def make_simple_corner(flat_samples, color="C0", fig=None, axarr=None, npar=3, figsize=(8, 8), wspace=0.2, hspace=0.2,
                       axis_labels=("log10 M", "c", "b"), limits=((14, 15), (0, 8), (0, 8)), **kwargs):
    if fig is None:
        fig, axarr = plt.subplots(nrows=npar, ncols=npar, figsize=figsize, sharex=False, sharey=False)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

    for i in np.arange(npar):
        axarr[-1, i].set_xlabel(axis_labels[i], fontsize=12)
        for j in np.arange(npar):
            if j > i:
                axarr[i, j].axis('off')

    for i in np.arange(npar):
        axarr[-1, i].set_xlabel(axis_labels[i], fontsize=12)
        axarr[i, 0].set_ylabel(axis_labels[i], fontsize=12)
        for j in np.arange(npar):
            if j < i:
                axarr[i, j].set_xlim(limits[j])
                axarr[i, j].set_ylim(limits[i])

                nbins = 40
                bins = (
                    np.linspace(limits[j][0], limits[j][1], nbins),
                    np.linspace(limits[i][0], limits[i][1], nbins),
                )
                xcens = bins[0][:-1] + np.diff(bins[0]) / 2
                ycens = bins[1][:-1] + np.diff(bins[1]) / 2

                xx, yy = np.meshgrid(xcens, ycens)
                xx = xx.T
                yy = yy.T

                tmp = flat_samples

                _counts = np.histogram2d(tmp[:, j], tmp[:, i], bins=bins)[0]
                counts = ndimage.gaussian_filter(_counts, sigma=1.0, order=0)
                mx = counts.max()
                axarr[i, j].contour(xx, yy, counts,levels=[mx*0.05, mx*0.2, mx*0.5, mx*0.8],
                                    linewidths=0.9, colors=color, **kwargs)


    for i in np.arange(npar):
        axarr[i, i].set_xlim(limits[i])
        bins = np.linspace(limits[i][0], limits[i][1], 40)
        tmp = flat_samples
        axarr[i, i].hist(tmp[:, i], histtype="step", color=color, density=True, bins=bins, **kwargs)


    axarr[0, 0].set_ylabel("p.d.f.")
    return fig, axarr



def contprep(data, clevels=(0.68, 0.95), sample=5000, weights=None, **kwargs):
    """
    Prepares contour levels based on passsed dataset

    Parameters
    ----------
    data : array with shape (N, 2)
        input data
    clevels : list
        probability values for contours
    sample : int
        number of samples to draw from data
    weights : None or np.array
        weights for each entry of data, uniform by default

    Returns
    -------
    tuple of np.arrays, tuple of ints
         (xx, yy, kk), levels
    """

    if weights is None:
        weights = np.ones(len(data))
    weights /= weights.sum()

    subsample = data[np.random.choice(np.arange(len(data)), int(sample), p=weights, replace=True), :]

    allgrid = kde_smoother_2d(subsample, **kwargs)

    tas = [conf2d(clevel, allgrid[0], allgrid[1], allgrid[2])[0] for clevel in  np.sort(clevels)[::-1]]

    return allgrid, tas


def conf1d(pval, grid, vals, res=500, etol=1e-2, **kwargs):
    """
    Calculates cutoff values for a given percentile for 1D distribution, Requires evenly spaced grid!

    Parameters:
    -----------
    pval : float
        probability to be contained within interval
    grid : np.array
        grid
    vals : np.array
        value of the p.d.f at given gridpoint
    res : float
        resolution of the percentile search

    Returns
    -------
    float, float
        cutoff value, actual percentile
    """

    area = np.mean(np.diff(grid))
    assert (np.sum(vals*area) - 1.) < etol, 'Incorrect normalization!!!'

    mx = np.max(vals)

    tryvals = np.linspace(mx, 0.0, res)
    pvals = np.array([np.sum(vals[np.where(vals > level)] * area)
                      for level in tryvals])

    tind = np.argmin((pvals - pval)**2.)
    tcut = tryvals[tind]
    return tcut, pvals[tind]


def conf2d(pval, vals, res=500, etol=1e-2, **kwargs):
    """
    Calculates cutoff values for a given percentile for 2D distribution, Requires evenly spaced grid!

    Parameters:
    -----------
    pval : float
        probability to be contained within interval
    # xxg : np.array
    #     grid for the first parameter
    # yyg : np.array
    #     grid for the second parameter
    vals : np.array
        value of the p.d.f at given gridpoint
    res : float
        resolution of the percentile search

    Returns
    -------
    float, float
        cutoff value, actual percentile
    """

    # edge1 = xxg[0, :]
    # edge2 = yyg[:, 0]
    #
    # area = np.mean(np.diff(edge1)) * np.mean(np.diff(edge2))
    # assert (np.sum(vals*area) - 1.) < etol, 'Incorrect normalization!!!'
    #
    mx = np.max(vals)
    tryvals = np.linspace(mx, 0.0, res)
    pvals = np.array([np.sum(vals[np.where(vals > level)])
                      for level in tryvals])

    tind = np.argmin((pvals - pval)**2.)
    tcut = tryvals[tind]

    return tcut, pvals[tind]


def kde_smoother_1d(pararr, xlim=None, num=100, pad=0):
    """
    Creates a smoothed histogram from 1D scattered data

    Parameters
    ----------
    pararr : np.array with shape (Npoint, Npar)
        list of parameters shape
    xlim :  tuple or None
        x range of the grid
    num : int
        number of gridpoints on each axis
    pad : float
        padding to use if xlim is not specified

    Returns
    -------
    np.array, np.array
        xgrid, values for each point
    """
    # creating smoothing function
    kernel = stats.gaussian_kde(pararr)

    # getting boundaries
    if xlim is None:
        xlim = [np.min(pararr), np.max(pararr)]
        xpad = pad * np.diff(xlim)
        xlim[0] -= xpad
        xlim[1] += xpad
    # building grid
    xgrid = np.linspace(xlim[0], xlim[1], num)

    # evaluating kernel on grid
    kvals = kernel(xgrid)

    return xgrid, kvals


def kde_smoother_2d(pararr, xlim=None, ylim=None, num=100, pad=0.1):
    """
    Creates a smoothed histogram from 2D scattered data

    Parameters
    ----------
    pararr : np.array with shape (Npoint, Npar)
        list of parameters shape
    xlim :  tuple or None
        x range of the grid
    ylim :  tuple or None
        y range of the grid
    num : int
        number of gridpoints on each axis
    pad : float
        padding to use if xlim is not specified

    Returns
    -------
    np.array, np.array, np.array
        xgrid, ygrid, values for each point
    """
    # creating smoothing function
    kernel = stats.gaussian_kde(pararr.T)

    # getting boundaries
    if xlim is None:
        xlim = [np.min(pararr[:, 0]), np.max(pararr[:, 0])]
        xpad = pad * np.diff(xlim)
        xlim[0] -= xpad
        xlim[1] += xpad
    if ylim is None:
        ylim = [np.min(pararr[:, 1]), np.max(pararr[:, 1])]
        ypad = pad * np.diff(ylim)
        ylim[0] -= ypad
        ylim[1] += ypad

    # building grid
    xgrid = np.linspace(xlim[0], xlim[1], num)
    ygrid = np.linspace(ylim[0], ylim[1], num)
    xx, yy = np.meshgrid(xgrid, ygrid )
    grid_coords = np.append(xx.reshape(-1,1), yy.reshape(-1,1),axis=1)

    # evaluating kernel on grid
    kvals = kernel(grid_coords.T).reshape(xx.shape)

    return xx, yy, kvals


def corner(par_names, pars, par_edges, figsize=(6, 6), color='black', fig=None,
           axarr=None, mode="contour", cmap="gray_r", normed=True, fontsize=12,
           tick_list=None, clevels=(0.95, 0.68), grid=True, weights=None, lw=1, auto_transpose_pars=True, **kwargs):
    """
    Creates *NICE* corner plot (check if pars have the righ orientation, perhaps use pars.T)

    Example
    -------

    Simple 2D gaussian with marginals and joint distribution::

        import numpy as np
        import xpipe as xpipe

        # generate mock data
        means = [0, 0]
        cov = [[1., 0.4], [0.4, 1.]]
        pars = np.random.multivariate_normal(means, cov, size=int(1e4)).T

        # define names and edges
        par_names = ["$x$", "$y$"]
        par_edges = (np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))

        fig, axarr = xpipe.tools.visual.corner(par_names, pars, par_edges)


    Parameters
    ----------
    par_names : list of strings
        names of parameters as they should appear on figure
    pars : np.ndarray
        the DATA to show, (might have to transpose it!
    par_edges : list of tuples
        parameter limits
    figsize : tuple
        size of matplotlib figure
    color : str
        primary color to use
    fig : mpl.figure
        figure
    axarr : list of mpl.axes
        axarr
    mode : str
        hist or contour
    cmap : str
        matplotlib color map to use for histogram
    normed : bool
        normalize histograms across subplots
    fontsize : int
        fontsize
    tick_list : list of np.arrays
        list of tick positions for each axis
    clevels : list
        contour levels as contained probability
    weights : weights for each entry in pars

    Returns
    -------
    plt.figure, (plt.axis, ..., plt.axis)
        fig, axarr

    """
    if auto_transpose_pars:
        if pars.shape[0] > pars.shape[1]:
            pars = pars.T

    npars = len(par_names)
    if fig is None and axarr is None:
        fig, axarr = plt.subplots(nrows=npars, ncols=npars, sharex=False,
                                  sharey=False, figsize=figsize)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        axarr[0, 0].set_yticklabels([])

        # hiding upper triangle
        [[ax.axis('off') for ax in axrow[(i + 1):]]
         for i, axrow in enumerate(axarr)]

        # hiding xlabels
        [[ax.set_xticklabels([]) for ax in axrow]
         for i, axrow in enumerate(axarr[:-1, :])]
        [[ax.set_yticklabels([]) for ax in axrow[1:]]
         for i, axrow in enumerate(axarr[:, :])]

        # Adding the distribution of parameters
        [ax.set_xlabel(par, fontsize=fontsize)
         for (ax, par) in zip(axarr[-1, :], par_names)]
        [ax.set_ylabel(par, fontsize=fontsize)
         for (ax, par) in zip(axarr[1:, 0], par_names[1:])]

        [ax.tick_params(labelsize=8) for ax in axarr.flatten()]

    [axarr[i, i].hist(pars[i], bins=par_edges[i], color=color,
                      histtype='step', weights=weights, lw=lw, density=normed)
     for i in range(len(pars))]

    if tick_list is not None:
        [axarr[i, i].xaxis.set_ticks(tick_list[i])
         for i in range(len(pars))]

    [axarr[i, i].set_xlim((par_edges[i][0], par_edges[i][-1]))
     for i in range(len(pars))]

    for i, axrow in enumerate(axarr[1:]):
        for j, ax in enumerate(axrow[:i + 1]):

            if tick_list is not None:
                ax.xaxis.set_ticks(tick_list[j])
                ax.yaxis.set_ticks(tick_list[i + 1])

            if mode == "hist":
                counts, xedges, yedges, cax = ax.hist2d(pars[j], pars[i + 1],
                                                        bins=(par_edges[j], par_edges[i + 1]), cmap=cmap,
                                                        density=normed, weights=weights, **kwargs)
                ax.set_xlim((par_edges[j][0], par_edges[j][-1]))
                ax.set_ylim((par_edges[i + 1][0], par_edges[i + 1][-1]))

            elif mode == "contour":
                # xlim = (par_edges[j][0], par_edges[j][-1])
                # ylim = (par_edges[i + 1][0], par_edges[i + 1][-1])
                bins = (
                    par_edges[j],
                    par_edges[i + 1]
                )
                xcens = bins[0][:-1] + np.diff(bins[0]) / 2
                ycens = bins[1][:-1] + np.diff(bins[1]) / 2
                # print(xens.shape)
                xx, yy = np.meshgrid(xcens, ycens)
                xx = xx.T
                yy = yy.T

                # tmp = flat_samples

                _counts = np.histogram2d(pars[j], pars[i + 1], bins=bins)[0]
                counts = ndimage.gaussian_filter(_counts, sigma=1.0, order=0)
                counts = counts / counts.sum()


                levels = [conf2d(lev, counts)[0] for lev in clevels]
                # print(levels)
                # mx = counts.max()
                ax.contour(xx, yy, counts, levels=levels,
                                    linewidths=lw, colors=color, **kwargs)

                # params = np.vstack((pars[j], pars[i + 1])).T
                # xlim = (par_edges[j][0], par_edges[j][-1])
                # ylim = (par_edges[i + 1][0], par_edges[i + 1][-1])
                # # print(params.shape)
                # allgrid, tba = contprep(params, xlim=xlim, ylim=ylim, clevels=clevels, weights=weights, **kwargs)
                # # print(tba)
                # ax.contour(allgrid[0], allgrid[1], allgrid[2],
                #            levels=tba, colors=color)
                # ax.contourf(allgrid[0], allgrid[1], allgrid[2],
                #             levels=[tba[1], np.inf], colors=color, alpha=0.7)

    return fig, axarr
