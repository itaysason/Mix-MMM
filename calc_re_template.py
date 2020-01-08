from os.path import join
from src.utils import get_model, load_json
import numpy as np
import pandas as pd
import sys
#sys.path.insert(0, './utils')
from calc_re import comp_re_ours, comp_re_sigma, get_index


if __name__ == "__main__":
    #train on 2, test on 1 or vice versa
    part_ls = [[2,1],[1,2]]
    #downsize ratios
    ds_ls = [2, 5, 10]
    #active signatures for ov
    sig_ls = [1,3,5]
    raw_M = "/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/ov-all_sigma_sbs.tsv"
    raw_sbs = pd.read_csv(raw_M,sep=',')
    #print(raw_sbs)
    cosmic_f ="data/signatures/COSMIC/cosmic-signatures.tsv"
    method = "Mix"
    if method == "Mix":
        for dl in ds_ls:
            # load shuffled index file
            raw_M = "data/"
            index_all = np.load("/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize/indices/OV-ds%d-indices.npy"%dl)
            for pl in part_ls:
                # get test index
                if pl[1] == 1:
                    index_list =  index_all[:len(index_all) // 2] 
                elif pl[1] == 2:
                    index_list = index_all[len(index_all) // 2:]
                path = join("/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize/models/OV-ds%d-part%d-parameters.json"%(dl, pl[0]))
                data = join("/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize/models/OV-ds%d-part%d-data.npy"%(dl, pl[1]))
                model=get_model(load_json(path))
                tmp = model.predict(np.load(data))
                # tmp[0] is the cluster membership, tmp[1] is the exposure matrix
                expo_np = tmp[1]
                our_re = comp_re_ours(raw_sbs, expo_np, index_list, sig_ls, cosmic_f)
                print("When method=%s, downsize ratio=%d, the RE of %s is %0.4f test on part %d data"%(method, dl,method, our_re, pl[1]))