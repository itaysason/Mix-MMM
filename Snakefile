from os.path import join
##################SELECT YOUR ONCOTREE CODE
#please refer to data/processed/oncotype_counts.txt for oncotree_code and number of samples associated with this code.
#for a given oncoTree code, you can go to http://oncotree.mskcc.org/#/home to find its full name.
OT_CODE='BLCA'
##################
RAW_DATA_URL="https://obj.umiacs.umd.edu/mutation-signature-explorer/mutations/cBioPortal/processed/counts/counts.cBioPortal-msk-impact-2017_%s_6800765.TARGETED.SBS-96.tsv"%OT_CODE
META_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/msk_impact_2017/data_clinical_sample.txt"
###################LOCAL DIRS
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
UTILS_DIR = 'utils'
SRC_DIR = 'src'
###################LOCAL FILES
MSK_RAW_DATA_FILE = join(RAW_DIR,"counts.cBioPortal-msk-impact-2017_%s_6800765.TARGETED.SBS-96.tsv"%OT_CODE)
MSK_META_FILE = join(RAW_DIR, "data_clinical_sample.txt")
MSK_COUNT_FILE= join(PROCESSED_DIR, '%s_counts.npy'%OT_CODE)
MSK_VIEW_FILE = join(PROCESSED_DIR, 'oncotype_counts.txt')
MSK_ID_FILE= join(PROCESSED_DIR, '%s_sample_id.txt'%OT_CODE)
###################SCRIPTS
MSK_PRE_SRC = join(UTILS_DIR, 'msk_preprocess.py')
MSK_VIEW_SRC = join(UTILS_DIR, 'msk_overview.py')
#BOX_SRC = join(UTILS_DIR, 'box_plot.py')

rule all:
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
	shell:
		"python {MSK_PRE_SRC} -rd {RAW_DIR} -ot {OT_CODE} -od {PROCESSED_DIR}"

rule view_msk:
	input:
		MSK_META_FILE
	output:
		MSK_VIEW_FILE
	shell:
		"python {MSK_VIEW_SRC} -rf {MSK_META_FILE} -of {MSK_VIEW_FILE}"


rule download_msk_raw:
	output:
		MSK_RAW_DATA_FILE
	shell:
		"wget -O {MSK_RAW_DATA_FILE} {RAW_DATA_URL}"

rule download_msk_meta:
    output:
        MSK_META_FILE
    shell:
        "wget -O {MSK_META_FILE} {META_URL}"
