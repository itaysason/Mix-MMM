from os.path import join
##################SELECT YOUR ONCOTREE CODE
#please refer to data/processed/oncotype_counts.txt for oncotree_code and number of samples associated with this code.
#for a given oncoTree code, you can go to http://oncotree.mskcc.org/#/home to find its full name.
OT_CODE=["LUAD", "IDC", "COAD", "PRAD","PAAD", "BLCA", "GBM", "CCRCC","SKCM", "ILC", "LUSC", "STAD","READ", "CUP", "GIST", "HGSOC","IHCH", "ESCA"]
#OT_CODE=["COAD", "IDC"]
##################
RAW_DATA_URL=expand("https://obj.umiacs.umd.edu/mutation-signature-explorer/mutations/cBioPortal/processed/counts/counts.cBioPortal-msk-impact-2017_{ot_code}_6800765.TARGETED.SBS-96.tsv",ot_code=OT_CODE)
META_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/msk_impact_2017/data_clinical_sample.txt"
###################LOCAL DIRS
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
UTILS_DIR = 'utils'
SRC_DIR = 'src'
###################LOCAL FILES
MSK_RAW_DATA_FILE =expand(join(RAW_DIR,"counts.cBioPortal-msk-impact-2017_{ot_code}_6800765.TARGETED.SBS-96.tsv"), ot_code=OT_CODE)
MSK_META_FILE = join(RAW_DIR, "data_clinical_sample.txt")
MSK_COUNT_FILE= expand(join(PROCESSED_DIR, '{ot_code}_counts.npy'), ot_code=OT_CODE)
MSK_VIEW_FILE = join(PROCESSED_DIR, 'oncotype_counts.txt')
MSK_ID_FILE= expand(join(PROCESSED_DIR, '{ot_code}_sample_id.txt'), ot_code=OT_CODE)
###################SCRIPTS
MAIN_SRC ="main.py"
MSK_PRE_SRC = join(UTILS_DIR, 'msk_preprocess.py')
MSK_VIEW_SRC = join(UTILS_DIR, 'msk_overview.py')
#BOX_SRC = join(UTILS_DIR, 'box_plot.py')
###################CONFIGURATIONS
#dataset
DATASET="MSK-ALL"
#number of signatures
NUM_SIG=4
#number of clusters
NUM_CLUSTERS=4
#number of fold
NUM_FOLD=5
#use cosmic or not
COSMIC_BOOL=1
#max iterations
MAX_ITER=1
#random seed
SEED=1234

rule train:
	shell:
		"python {MAIN_SRC} train_model --dataset {DATASET} \
--num_clusters {NUM_CLUSTERS} --use_cosmic {COSMIC_BOOL} --num_signatures {NUM_SIG} \
--random_seed {SEED} --max_iterations {MAX_ITER}"

rule samplecv:
	run:
		for fd in range(NUM_FOLD):
			shell("python {MAIN_SRC} sampleCV --dataset {DATASET} \
--num_clusters {NUM_CLUSTERS} --use_cosmic {COSMIC_BOOL} --num_folds {NUM_FOLD} \
--fold %d --num_signatures {NUM_SIG} --random_seed {SEED} --max_iterations {MAX_ITER}"%fd)

rule msk_all:
	input:
		MSK_COUNT_FILE,
		MSK_ID_FILE,
		MSK_VIEW_FILE

rule process_msk:
	input:
		MSK_RAW_DATA_FILE
	output:
		MSK_COUNT_FILE,
		MSK_ID_FILE
	run:
		for j in range(len(MSK_RAW_DATA_FILE)):
			shell("python {MSK_PRE_SRC} -rd {RAW_DIR} -ot {OT_CODE[%d]} -od {PROCESSED_DIR}"%j)

rule view_msk:
	input:
		MSK_META_FILE
	output:
		MSK_VIEW_FILE
	shell:
		"python {MSK_VIEW_SRC} -rf {MSK_META_FILE} -of {MSK_VIEW_FILE}"


rule download_msk_raw:
#download multiple files at the same time
	output:
		MSK_RAW_DATA_FILE
	run:
		for i in range(len(OT_CODE)):
			shell("wget -O {MSK_RAW_DATA_FILE[%d]} {RAW_DATA_URL[%d]}"%(i,i))

rule download_msk_meta:
    output:
        MSK_META_FILE
    shell:
        "wget -O {MSK_META_FILE} {META_URL}"
