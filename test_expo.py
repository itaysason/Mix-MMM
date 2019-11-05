from src.utils import get_model, load_json
from os.path import join
import numpy as np
import sys
sys.path.insert(0, './utils')
from calc_re import comp_re_ours, comp_re_sigma, get_index

if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    raw_M = join(msk_dir, "ov-all_sigma_sbs.tsv")
    cosmic = join(msk_dir, "cosmic-signatures.tsv")
    method = "nnls"
    part_ls = [[2,1],[1,2]]
    ds_ls = [2, 5, 10]
    #for ov
    sig_ls = [1,3,5]
    
    if method == "Mix":
        for dl in ds_ls:
            index_all = np.load(join(msk_dir,"mix_downsize/indices/OV-ds%d-indices.npy"%dl))
            for pl in part_ls:
                #index_all = np.load(join(msk_dir,"mix_downsize/indices/OV-ds%d-indices.npy"%dl))
                if pl[1] == 1:
                    index_list =  index_all[:len(index_all) // 2] 
                elif pl[1] == 2:
                    index_list = index_all[len(index_all) // 2:]
                path = join(msk_dir,"mix_downsize/models/OV-ds%d-part%d-parameters.json"%(dl, pl[0]))
                data = join(msk_dir,"mix_downsize/models/OV-ds%d-part%d-data.npy"%(dl, pl[1]))
                model=get_model(load_json(path))
                tmp = model.predict(np.load(data))
                np.save(join(msk_dir, "mix_downsize/models/OV-ds%d-part%d-exposure.npy"%(dl,pl[1])),tmp[1],allow_pickle=True)
                expo_np = tmp[1]
                re_ours = comp_re_ours(raw_M, expo_np, index_list, sig_ls, cosmic)
                print("When method=%s, downsize ratio=%d, the RE of sigma is %0.4f on part %d data"%(method, dl, re_ours, pl[1]))
    elif method == "SigMA":
        sig_part = [1,2]
        for dl in ds_ls:
            index_all = np.load(join(msk_dir,"mix_downsize/indices/OV-ds%d-indices.npy"%dl))
            for s in sig_part:
                ds_sbs = join(msk_dir, "ov-downsize%s_sigma_sbs_part%d.tsv"%(dl,s))
                sigma_out = join(msk_dir,"out-ov-downsize%s_sigma_sbs_part%d.tsv"%(dl,s))
                if s == 1:
                    index_list =  index_all[:len(index_all) // 2] 
                elif s == 2:
                    index_list = index_all[len(index_all) // 2:]
                #print(index_list)
                #print(index_all)
                re_sigma, exp_3 = comp_re_sigma(raw_M, sigma_out, index_list, cosmic)
                np.save(join(msk_dir, "mix_downsize/sigma_downsize%d_sig3_part%d.npy"%(dl,s)), exp_3, allow_pickle=True)
                print("When method=%s, downsize ratio=%d, the RE of sigma is %0.4f on part %d data"%(method, dl, re_sigma, s))
    elif method == "nnls":
        sig_part = [1,2]
        for dl in ds_ls:
            index_all = np.load(join(msk_dir,"mix_downsize/indices/OV-ds%d-indices.npy"%dl))
            for s in sig_part:
                #index_all = np.load(join(msk_dir,"mix_downsize/indices/OV-ds%d-indices.npy"%dl))
                if s == 1:
                    index_list =  index_all[:len(index_all) // 2] 
                elif s == 2:
                    index_list = index_all[len(index_all) // 2:]
                expo_np = np.load(join(msk_dir, "mix_downsize/models/nnls-OV-ds%d-part%d-exposure.npy"%(dl, s)))
                re_ours = comp_re_ours(raw_M, expo_np, index_list, sig_ls, cosmic)
                print("When method=%s, downsize ratio=%d, the RE of sigma is %0.4f on part %d data"%(method, dl, re_ours, s))
