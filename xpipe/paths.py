"""
Loads project path from user config files
"""

import os
import yaml
import glob
import numpy as np

# TODO replace print with logging

default_inputs_suffix = 'settings/default_inputs.yml'
inputs_suffix = 'settings/inputs.yml'


###################################################################
# readers

def get_poject_path(user_cfg):
    """Loads the absolute path to the cluster pipeline"""
    path = os.path.expanduser('~') + '/' + user_cfg
    with open(path)as file:
         cfg = yaml.safe_load(file)
    return cfg['project_path']


def read_yaml(cfg):
    """read directory tree form config file"""
    with open(cfg)as file:
         cfg = yaml.safe_load(file)
    return cfg


def get_pdf_flist(params):
    ppaths = None
    if params['pdf_paths'] is not None:
        ppaths = np.sort(glob.glob(params['pdf_paths']))
    return ppaths


###################################################################
# modeset

def assign_mode(params):
    _devmode = True
    if params['mode'] == 'full':
        _devmode = False
    return _devmode


def print_mode(params):
    _devmode = True
    if params['mode'] == 'full':
        _devmode = False

    print '\n***********************\n'
    if _devmode:
        print 'running in test / development mode'
    else:
        print 'running in full mode'

    print '\n***********************\n'


###################################################################
# filepath manipulation

def expand_paths(dct, root_path):
    fullpath = {}
    fullpath.update({'root': root_path})
    for key in dct.keys():
        fullpath.update({key: root_path + dct[key]})
    return fullpath


def get_dirpaths(params, project_path):
    paths_file = project_path + 'settings/default_paths.yml'
    _dirs = read_yaml(paths_file)

    custom_data_path = params['custom_data_path']
    if not custom_data_path:
        custom_data_path = project_path + 'data'
    _dirpaths = expand_paths(_dirs['data'], custom_data_path)
    return _dirpaths


def get_local_filenames(pths, dirs):
    fullpaths = {}
    for dkey in pths.keys():
        for item in pths[dkey]:
            if isinstance(pths[dkey][item], str):
                ftail = pths[dkey][item].split('/')[-1]
                fullpaths.update({item: dirs[dkey] + '/' + ftail})
            elif isinstance(pths[dkey][item], dict):
                idict = {}
                for key in pths[dkey][item].keys():
                    ftail = pths[dkey][item][key].split('/')[-1]
                    idict.update({key: dirs[dkey] + '/' + ftail})
                fullpaths.update({item: idict})

    return fullpaths


def get_fullpaths(params, project_path, default_inputs=True):

    # reading default input file urls
    _default_inputs_file = project_path + default_inputs_suffix
    _inputs_dict = read_yaml(_default_inputs_file)

    try:
        _inputs_file = project_path + inputs_suffix
        _inputs_dict.update(read_yaml(_inputs_file))
    except:
        pass

    fullpaths = get_local_filenames(_inputs_dict['local'], dirpaths)

    # check if shear_to_use is present in fullpaths
    if params['shear_to_use'] not in fullpaths.keys():
        raise KeyError(
            'Specified shearcat ("shear_to_use: ' + str(params['shear_to_use']) + '") is not in the local shearcats!')

    # xshear config file
    fullpaths.update({'xpath': project_path + 'submodules/xshear/bin/xshear'})

    return fullpaths

###################################################################
# updater for custom params file

def use_custom_params(path):
    global params, has_custom_specified
    if os.path.isfile(path):
        print 'updating params from: ' + path
        tmp_yaml = read_yaml(path)
        params.update(tmp_yaml)
        has_custom_specified = "custom_params_file" in tmp_yaml.keys()


def locate_params(pth):
    # print pth
    _pth = os.path.expanduser(pth)
    head, tail = os.path.split(pth)
    if head == "":
        _pth = project_path + "settings/" + tail
    return _pth


def _update_params(path):
    if path is not None:
        _path = locate_params(path)

        global devmode, dirpaths, fullpaths, fullurls, pdf_files

        use_custom_params(_path)
        devmode = assign_mode(params)
        dirpaths = get_dirpaths(params, project_path)
        fullpaths = get_fullpaths(params, project_path)
        pdf_files = get_pdf_flist(params)


def update_params(path):
    """Updates the dictionary with custom parameter file located at 'path' """
    _update_params(path)
    print_mode(params)


def _complete_params_path(fname):
    """adds project path to params file"""
    opath = fname
    head, tail = os.path.split(fname)
    if head == "":
        opath = project_path + "settings/" + fname
    return opath


def get_bin_settings(params, devmode):
    """Returns appropriate bin edges and the number of random points to use"""
    if devmode:
        param_bins = params['param_bins_dev']
        nrandoms = params['nrandoms']['dev']
    else:
        param_bins = params['param_bins_full']
        nrandoms = params['nrandoms']['full']
    keys = np.sort(param_bins.keys())
    param_bins = [param_bins[key] for key in keys]

    return param_bins, nrandoms


###################################################################
#
#   loading params
#
###################################################################

# reading the absolute path of the project
user_project_file = '.xpipe.yml'
project_path = get_poject_path(user_project_file)

# read parameter bins from config file
print "reading DEFAULTS from default_params.yml"

default_param_path = project_path + 'settings/default_params.yml'
params = read_yaml(default_param_path)

devmode = assign_mode(params)
dirpaths = get_dirpaths(params, project_path)
fullpaths = get_fullpaths(params, project_path, default_inputs=True)
pdf_files = get_pdf_flist(params)

# READING custom params files
has_custom_specified = "custom_params_file" in params.keys()
while has_custom_specified and params["custom_params_file"] is not None:
    custom_param_path = _complete_params_path(params["custom_params_file"])
    if os.path.isfile(custom_param_path):
        _update_params(custom_param_path)
    else:
        break


# TODO add effective re-initialization of params for API mode


def set_params(**kwargs):
    global devmode, dirpaths, fullpaths, fullurls, pdf_files

