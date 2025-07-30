'''
This script allows visualize Variant loci with Cuban based on a CSV file.
'''

import argparse
import pathlib
import logzero
import logging
import pandas as pd
import cuban_lib.visualize as visualize
import cuban_lib.utils as utils
from pandarallel import pandarallel
import matplotlib.pyplot as plt
import os
import json

plt.style.use('ggplot')

def argparser():
    '''
    Parse CLIs
    '''
    parser = argparse.ArgumentParser("Visualizes Variant loci using Coolbox.")
    parser.add_argument('-sv', '--variants',
                        required=True, help='Input variants in BED format.')
    parser.add_argument('-b', '--bams', nargs='+', required=True, help='Path to the BAM files.')
    parser.add_argument('-ilp', '--illumina_path', required=True, help='Template path to the Illumina BAM file.')
    parser.add_argument('-pbp', '--lrs_path', required=True, help='Template path to the LRS BAM file.')
    parser.add_argument('-c', '--chroms', required=False, nargs='+', default=[], help='list of chromosomes to restrict the analysis.')
    parser.add_argument('-s', '--sample', required=True, help='Sample ID.')
    parser.add_argument('-p', '--pedigree', required=True, help='Cohort Pedigree file.')
    parser.add_argument('-r', '--repeats', help='Repeat tsv file.')
    parser.add_argument('-t', '--threads', required=True, help='Number of threads.')
    parser.add_argument('-e', '--exome_regions', required=False, help='BED file with regions targeted for sequencing.', default=None)
    parser.add_argument('-o', '--output_dir', help='Output directory for Cuban Figures.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Logging level.')
    args = parser.parse_args()
    return args

def plot_variant_loci_cuban(variant, samples, output_dir, rep_df, cov_dict):
    '''
    Generates a Cuban Figure for the variant loci.
    '''
    if not os.path.exists(output_dir / f'{variant["ID"]}.png'):
        logzero.logger.info(
            f'Saving Cuban Snapshot for {variant["ID"]}...')
        
        # Add coverage for the current chromosome
        for sample in samples:
            samples[sample]['baseline_cov'] = cov_dict[sample][variant['CHR']]
        # dont plot if the variants is larger than 2mb
        if variant['SIZE'] > 2000000:
            logzero.logger.info(f'Skipping {variant["ID"]} because it is larger than 2mb')
        else:
            visualize.cuban(samples, rep_df, variant['TYPE'],variant["CHR"], variant["START"], variant["END"],  padding=max(1500,round(int(variant['SIZE'])/10)), collapse_ins=False, window=100, sv_len=variant['SIZE'], outfile=output_dir / f'{variant["ID"]}.png')


def add_family_member_to_cuban_dict(row, index_sample, illumina_bam, lrs_bam, coverage_dict, family_ped_df, target_regions, cuban_sample_dict, chromosomes) -> dict:
    '''Adds sample information to the cuban dictionary based on available techs'''
    family_member = row['Name']
    if family_member!=index_sample:
        # check if illumina bam files for the family are available
        if os.path.isfile(f'{illumina_bam.replace(index_sample, family_member)}'):
            # assign family status
            if family_ped_df['Mother'].values[0] == family_member:
                family_status = 'mother'
            elif family_ped_df['Father'].values[0] == family_member:
                family_status = 'father'
            else:
                family_status = 'NA'
            coverage_dict[f'{family_member} - Illumina'] = {chrom:utils.compute_baseline_cov(illumina_bam.replace(index_sample, family_member), chrom, n=50, target_regions=target_regions) for chrom in chromosomes}
            cuban_sample_dict[f'{family_member} - Illumina'] = {'family_status':family_status, 'disease_status':row['Disease Status'], 'technology':'ill','bam_name':f'{illumina_bam.replace(index_sample, family_member)}'}
        else:
            logzero.logger.info(f'No Illumina BAM file available for {family_member}')
        # check if LRS bam files for the family are available
        if os.path.isfile(f'{lrs_bam.replace(index_sample, family_member)}'):
            logzero.logger.info(f'{lrs_bam.replace(index_sample, family_member)}')
            # assign family status
            if family_ped_df['Mother'].values[0] == family_member:
                family_status = 'mother'
            elif family_ped_df['Father'].values[0] == family_member:
                family_status = 'father'
            else:
                family_status = 'NA'
            coverage_dict[f'{family_member} - LRS'] = {chrom:utils.compute_baseline_cov(lrs_bam.replace(index_sample, family_member), chrom, n=50, target_regions=target_regions) for chrom in chromosomes}
            cuban_sample_dict[f'{family_member} - LRS'] = {'family_status':family_status, 'disease_status':row['Disease Status'], 'technology':'ill','bam_name':f'{lrs_bam.replace(index_sample, family_member)}'}
        else:
            logzero.logger.info(f'No LRS BAM file available for {family_member}')
    return cuban_sample_dict

def main():
    # read cli
    args = argparser()

    if args.verbose:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)

    variant_df = pd.read_csv(args.variants, header=0, index_col=None, sep=',')

    if not args.chroms:
        chromosomes = ['chr'+str(i) for i in range(1,23)] + ['chrX']
    else:
        chromosomes = args.chroms

    # check which bam files are available
    if len(args.bams) == 2:
        illumina_bam = args.bams[0]
        lrs_bam = args.bams[1]
    else:
        if 'mm2' in args.bams[0]:
            illumina_bam = None
            lrs_bam = args.bams[0]
        else:
            illumina_bam = args.bams[0]
            lrs_bam = None      
    
    # if provided read target regions
    if args.exome_regions is not None:
        target_regions = pd.read_csv(args.exome_regions, sep='\t', names=['chr','start','end'], dtype={'chr':str,'start':int,'stop':int},index_col=False)
    else:
        target_regions = args.exome_regions

    if not variant_df.empty:
        # parse pedigree data
        pedigree_df = pd.read_csv(args.pedigree, sep='\t', header=0, dtype={'Name':str,'Family':str,'Mother':str,'Father':str}, index_col=None)

        # subset pedigree information for the sample's family
        sample_ped_df = pedigree_df.loc[pedigree_df['Name']==args.sample]
        family = sample_ped_df['Family'].values[0]
        family_df = pedigree_df.loc[pedigree_df['Family']==family]

        # compute baseline coverage for the index sample    
        logzero.logger.info('Computing base-line coverages')
        coverage_dict = {}
        if illumina_bam:
            coverage_dict.update({f'{args.sample} - Illumina':{chrom:utils.compute_baseline_cov(illumina_bam, chrom, n=100, target_regions=target_regions) for chrom in chromosomes}})
        if lrs_bam:
            coverage_dict.update({f'{args.sample} - LRS':{chrom:utils.compute_baseline_cov(lrs_bam, chrom, n=100, target_regions=target_regions) for chrom in chromosomes}})
        
        # create sample dictionary based on the pedigree information
        if not lrs_bam and illumina_bam:
            logzero.logger.info('Illumina only provided!')
            cuban_sample_dict = {f'{args.sample} - Illumina': {'family_status' : 'index',
                            'disease_status' : sample_ped_df['Disease Status'].values[0],
                            'technology':'ill',
                            'bam_name' : illumina_bam}}
        elif lrs_bam and not illumina_bam:
           logzero.logger.info('LRS only provided!')
           cuban_sample_dict = {f'{args.sample} - LRS': {'family_status' : 'index',
                            'disease_status' : sample_ped_df['Disease Status'].values[0],
                            'technology':'pb',
                            'bam_name' : lrs_bam}}
        elif lrs_bam and illumina_bam:
           logzero.logger.info('Both Illumina and LRS provided!')
           cuban_sample_dict = {f'{args.sample} - Illumina': {'family_status' : 'index',
                            'disease_status' : sample_ped_df['Disease Status'].values[0],
                            'technology':'ill',
                            'bam_name' : illumina_bam},
                            f'{args.sample} - LRS': {'family_status' : 'index',
                            'disease_status' : sample_ped_df['Disease Status'].values[0],
                            'technology':'pb',
                            'bam_name' : lrs_bam},
                            }         
           
        for index, row in family_df.iterrows():
            cuban_sample_dict = add_family_member_to_cuban_dict(row, args.sample, args.illumina_path, args.lrs_path, coverage_dict, sample_ped_df, target_regions, cuban_sample_dict, chromosomes)

        # Load baseline coverage for control samples
        with open(pathlib.Path(__file__).parent.resolve() / 'resources/baseline_cov_ill.json', 'r') as f:
            coverage_dict['HG002 - Illumina'] = json.load(f)
        with open(pathlib.Path(__file__).parent.resolve() / 'resources/baseline_cov_pb.json', 'r') as f:
            coverage_dict['HG002 - PacBio HiFi'] = json.load(f)
                    
        # Samples from control
        control_samples = {'HG002 - Illumina' : {'family_status' : 'control',
                                                'disease_status' : 'unaffected',
                                                'technology' : 'ill',
                                                'bam_name' : str(pathlib.Path(__file__).parent.resolve() / 'HG002.illumina.bam')},
                            'HG002 - PacBio HiFi' : {'family_status' : 'control',
                                                    'disease_status' : 'unaffected',
                                                    'technology' : 'pb',
                                                    'bam_name' : str(pathlib.Path(__file__).parent.resolve() / 'HG002.pacbio.bam'),}}
        cuban_sample_dict.update(control_samples)

        # Load repeat data as dataframe
        repeat_df = pd.read_csv(args.repeats, sep='\t')

        output_path = pathlib.Path(args.output_dir)

        logzero.logger.info('Generating Cuban Visualizations..')
        kwargs = {'samples': cuban_sample_dict, 'output_dir': output_path, 'rep_df':repeat_df, 'cov_dict':coverage_dict}

        # Initiliase parallel processing
        pandarallel.initialize(progress_bar=False, nb_workers=int(args.threads))

        # Create Cuban plots for each variant
        variant_df.parallel_apply(plot_variant_loci_cuban, axis=1, **kwargs)


if __name__ == '__main__':
    main()