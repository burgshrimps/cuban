import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
import os
from joblib import Parallel, delayed
from glob import glob

from lib.visualize import plot_breakpoints_ill, plot_breakpoints_ill_pb
from lib.utils import add_comma_to_pos

plt.style.use('ggplot')


def plotqc(CHUNK, ERR):
    """ Generates plots for each SV in the dataframe.

    param index: index of dataframe to plot """

    QCDIR = f'/project/Variation/detection/svc/curation/GRCh38/{SAMPLE}/{DATE}/{TYPE}/{ERR}/{CHUNK}'
    QCFILE = f'{DATE}_{ERR}_{TYPE}_{CHUNK}.tsv'

    if not os.path.exists(f'{QCDIR}/images'):
        os.makedirs(f'{QCDIR}/images')

    df = pd.read_csv(f'{QCDIR}/{QCFILE}', sep='\t')
    for i in tqdm(range(df.shape[0])):
        svid = df.loc[i, 'id']
        chrom = df.loc[i, 'chrom']
        start = df.loc[i, 'start']
        end = df.loc[i, 'end']
        title = svid + ' ' + chrom + ':' + add_comma_to_pos(start) + '-' + add_comma_to_pos(end)
        
        if TYPE == 'INS' or TYPE == 'DUP':
            collapse = False
        else:
            collapse = True
            
        plot_breakpoints_ill_pb(BAM_ILL, BAM_PB, REPEATS, chrom, start, end, collapse_ins=collapse, window=100, title=title, outfile=f'{QCDIR}/images/{svid}.png')


DATE = sys.argv[1]
SAMPLE = sys.argv[2]
SAMPLE_ = SAMPLE.replace('-', '_')
TYPE = sys.argv[3]
ERR = sys.argv[4]

BAM_ILL = '/project/Variation/detection/svc/aligned_bam/HG002/mgi/GRCh38.bwa_mem.pe.sorted.mdup.recal.bam'
BAM_PB = '/project/Dicast/GIAB_data/PacBio_SequelII_CCS_11kb_HiFi/HG002_GRCh38.haplotag.10x.bam'
REPEATS = '/project/Dicast/reference/annotation/hg38_repeatmasker.tsv'

QCDIR = f'/project/Variation/detection/svc/curation/GRCh38/{SAMPLE}/{DATE}/{TYPE}/{ERR}'
CHUNKS = [path.split('/')[-1] for path in glob(f'{QCDIR}/*')]

Parallel(n_jobs=len(CHUNKS))(delayed(plotqc)(chunk, err) for chunk, err in zip(CHUNKS, [ERR]*len(CHUNKS)))