import numpy as np
import pandas as pd
import argparse
from os.path import join

"""
for a given step, produce Sig3+/- status
"""

def main(args):
    # read skeleton
    skeleton_df = pd.read_csv(args.skl_df, sep='\t')
    skeleton_id = skeleton_df['id'].tolist()
    # read id in mix
    ov_df = pd.read_csv(args.ov_idf, sep='\t')
    ov_id=[items[:12].lower() for items in ov_df.values[:,0].tolist()]

    #only choose common ones
    common_ls=list(set(skeleton_id).intersection(ov_id))
    print("we have %d common ids"%len(common_ls))
    common_df = skeleton_df.loc[skeleton_df['id'].isin(common_ls)]
    common_id = common_df['id'].tolist()

    # read mix exposures
    expo_ls = ['assignment', 'expected_exposure', 'hard_cluster_exposure', 'soft_cluster_exposure']
    for ep in expo_ls:
        now_f = join(args.mix_dir, ep+'.npy')
        now_np = np.load(now_f, allow_pickle=True)
        #normalize by row
        now_np = now_np/now_np.sum(axis=1)[:,np.newaxis]
        exp_3 = now_np[:,1].tolist()
        bina_exp3=[]
        for ex in exp_3:
            if ex < float(args.threshold):
                bina_exp3.append(0)
            else:
                bina_exp3.append(1)
        exp_dict=dict(zip(common_id, bina_exp3))
        brca_dict=dict(zip(common_id, list(common_df['brcaness'])))
        now_bina=[]
        for ci in common_id:
            exp3_stat=exp_dict[ci]
            brca_stat=brca_dict[ci]
            if brca_stat==1:
                now_bina.append('BRCAness')
            else:
                if exp3_stat==1:
                    now_bina.append('Sig3+')
                else:
                    now_bina.append('Sig3-')
        common_df['exp3']=now_bina
        surv_f=join(args.surv_dir, ep+'_surv.tsv')
        common_df.to_csv(surv_f, sep='\t')



if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-oif', '--ov_idf')
    parser.add_argument('-md', '--mix_dir')
    parser.add_argument('-sd', '--skl_df')
    parser.add_argument('-thre', '--threshold')
    parser.add_argument('-sud', '--surv_dir')
    args=parser.parse_args()
    main(args)

