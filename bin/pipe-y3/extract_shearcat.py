import pandas as pd
import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import yaml
import fitsio


main_file_path = "/e/ocean1/users/vargatn/DESY3/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5"
main_file = h5py.File(main_file_path, mode = 'r')
fname_root = "/e/ocean1/users/vargatn/DES/DES_Y3A2_cluster/data/shearcat/"
fname_prefix = "y3_mcal_sompz_v3_unblind"

fmt = ['%d', '%d'] + 10 * ['%.6f']
sheared_names = ["unsheared", "_1p", "_1m", "_2p", "_2m"]
# sheared_names = ["unsheared", "sheared_1p", "sheared_1m", "sheared_2p", "sheared_2m"]
select_names = ["select", "select_1p", "select_1m", "select_2p", "select_2m"]
bin_names = ["bin1", "bin2", "bin3", "bin4"]

for ibin, bin_name in enumerate(bin_names):
    for ishear, sheared_name in enumerate(sheared_names):
        if ishear == 0:
            _sheared_name = ""
            ztag = sheared_name
            _name = ""
        else:
            _sheared_name = "_sheared" + sheared_name
            ztag = "sheared" + sheared_name
            _name = sheared_name

        oname = fname_root + fname_prefix + "_" + bin_name + _name + ".dat"
        print(oname)

        index = main_file["index/"+select_names[ishear] + "_" + bin_name][:]

        table = pd.DataFrame()

        # loading part 1
        _tmp = main_file["index/coadd_object_id"][:]
        table["coadd_objects_id"] = _tmp[index]
        print("loaded coadd_id")

        _tmp = main_file["catalog/sompz/" + ztag + "/cell_wide"][:]
        table["cell_id"] = _tmp[index]
        print("loaded cell_id")

        _tmp = main_file["catalog/gold/ra"][:]
        table["ra"] = _tmp[index]
        _tmp = main_file["catalog/gold/dec"][:]
        table["dec"]= _tmp[index]
        print("loaded ra, dec")

        #         print("reading metacal unsheared")
        _tmp = main_file["catalog/metacal/" + ztag + "/e_1"][:]
        table["e1"] = _tmp[index]
        _tmp = main_file["catalog/metacal/" + ztag + "/e_2"][:]
        table["e2"] = _tmp[index]
        print("loaded, e1, e2")
        if ishear == 0:
            _tmp = main_file["catalog/metacal/" + ztag + "/R11"][:]
            table["R11"] = _tmp[index]
            _tmp = main_file["catalog/metacal/" + ztag + "/R22"][:]
            table["R22"] = _tmp[index]
            print("loaded, R11, R22")
            _R_12 = _tmp[index]
            _tmp = main_file["catalog/metacal/" + ztag + "/R12"][:]
            _R_21 = _tmp[index]
            _tmp = main_file["catalog/metacal/" + ztag + "/R21"][:]
            table["R12"] = (_R_12 + _R_21) / 2.
            print("loaded, R12, R21")

        _tmp = main_file["catalog/metacal/" + ztag + "/weight"][:]
        table["weight"] = _tmp[index]

        placeholder = np.ones(len(table)) * 1.5
        table["z_mean"] = placeholder
        table["z_mc"] = placeholder

        table.to_hdf(oname.replace(".dat", ".h5"), key="data")
        table.to_csv(oname, index=False, sep=" ", header=False)
