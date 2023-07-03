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

    QCDIR = f'/confidential/tGenVar/scripts/tGenVar/dicast/training/curation/{REF}/{SAMPLE}/{DATE}/{TYPE}/{ERR}/{CHUNK}'
    QCFILE = f'{DATE}_{SAMPLE}_{ERR}_{TYPE}_{CHUNK}.tsv'

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
REF = sys.argv[3]
TYPE = sys.argv[4]
ERR = sys.argv[5]

BAM_ILL = f'/confidential/tGenVar/tech/illumina/snakemake_results/bam_{REF}/{SAMPLE_}/bwa_mem.pe.sorted.mdup.bam'
BAM_PB = f'/confidential/tGenVar/tech/pb/bam_MDtag_{REF}/pb.{SAMPLE_}.{REF}.bam'
REPEATS = f'/confidential/tGenVar/ref/{REF}/annotation/{REF}_repeatmasker.tsv'

QCDIR = f'/confidential/tGenVar/scripts/tGenVar/dicast/training/curation/{REF}/{SAMPLE}/{DATE}/{TYPE}/{ERR}'

CHUNKS = [path.split('/')[-1] for path in glob(f'{QCDIR}/*')]

Parallel(n_jobs=len(CHUNKS))(delayed(plotqc)(chunk, err) for chunk, err in zip(CHUNKS, [ERR]*len(CHUNKS)))