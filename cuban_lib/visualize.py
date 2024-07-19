import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from matplotlib import cm
import matplotlib.lines as mlines
import numpy as np
import pysam
import random 
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

from cuban_lib.utils import compute_aln_matrix, pad_alignment_matrices, compute_cov_df, compute_rep_df, get_variant_neighbourhood, compute_isize_orientation_dict, add_comma_to_pos

mpl.rcParams['agg.path.chunksize'] = 1000000
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
plt.style.use('ggplot')


rep_y_pos_map = {'LINE' : (-6, '#6ac0b7'),
                 'SINE' : (-14, '#b7954b'),
                 'LTR' : (-22, '#f0b6a0'),
                 'DNA' : (-30, '#5066a2'),
                 'Simple_repeat' : (-38, '#504669'),
                 'Satellite' : (-46, '#df624c'),
                 'Low_complexity' : (-54, '#61856b'),
                 'Retroposon' : (-62, '#2f7155')}


def add_splitread_overlay(aux_dict: dict, aln_matrix: np.array, ax: plt.axis, offset: int=0):
    """ Adds split read overlay to the plot.

    Args:
        aux_dict (dict): Axiliary dictionary indicating which read is split
        aln_matrix (np.array): Alignment matrix
        ax (plt.axis): Axis to plot on
        offset (int, optional): Offset. Defaults to 0.
    """    

    for idx in aux_dict['split_idx']:
        try:
            start = np.where(aln_matrix[idx] >= 4)[0][0]
            end = np.where(aln_matrix[idx] >= 4)[0][-1]
            ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx - 0.5),end-start+1, 0.9, hatch='||', fill=False, snap=False, linewidth=0.5, edgecolor='black'))
        except IndexError:
            pass


def add_disco_overlay(aux_dict: dict, aln_matrix: np.array, ax: plt.axis, orient: str, offset: int=0):
    """ Adds discordant read pair overlay to the plot.

    Args:
        aux_dict (dict): Axiliary dictionary indicating which read is discordant
        aln_matrix (np.array): Alignment matrix
        ax (plt.axis): Axis to plot on
        orient (str): Orientation of the discordant read pair (rr, ff, rf)
        offset (int, optional): Offset. Defaults to 0.
    """    
    
    for idx in aux_dict['discordant_idx_' + orient]:
        try:
            start = np.where(aln_matrix[idx] == 0)[0][0]
            end = np.where(aln_matrix[idx] == 0)[0][-1]
            if orient == 'rr':
                ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx-0.5),end-start+1, 1, hatch='\\\\',fill=False, snap=False, linewidth=0.5, edgecolor='sandybrown', alpha=1))
            elif orient == 'ff':
                ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx-0.5),end-start+1, 1, hatch='//',fill=False, snap=False, linewidth=0.5, edgecolor='cadetblue', alpha=1))
            elif orient == 'rf':
                ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx-0.5),end-start+1, 1, hatch='\\\\',fill=False, snap=False, linewidth=0.5, edgecolor='midnightblue', alpha=1))
            elif orient == 'tx':
                ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx-0.5),end-start+1, 1, hatch='//',fill=False, snap=False, linewidth=0.5, edgecolor='#df624c', alpha=1))
        except IndexError:
            pass


def add_mapq_overlay(aux_dict: dict, aln_matrix: np.array, ax: plt.axis, offset: int=0):
    """ Adds overlay to indicate low mapping quality reads.

    Args:
        aux_dict (dict): Axiliary dictionary indicating which read has low mapping quality
        aln_matrix (np.array): Alignment matrix
        ax (plt.axis): Axis to plot on
        offset (int, optional): Offset. Defaults to 0.
    """    

    for idx in aux_dict['low_mapq_idx']:
        try:
            start = np.where(aln_matrix[idx] >= 0)[0][0]
            end = np.where(aln_matrix[idx] >= 0)[0][-1]
            ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx-0.5),end-start+1,0.9,fill=True, snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
        
        
def gather_data_ill(bam_name: str, chrom: str, start: int, end: int, window: int, padding: int, collapse_ins: bool, rep_df: pd.DataFrame) -> dict:
    """ Gathers relevant sequencing data for one sample for a given region.

    Args:
        bam_name (str): Filename of the BAM file
        chrom (str): Chromosome
        start (int): Start position
        end (int): End position
        window (int): Window around each breakpoint to collect CIGAR string information for.
        padding (int): Padding around the SV to compute coverage.
        collapse_ins (bool): Collapse insertions into one base.
        rep_df (pd.DataFrame): Repeat dataframe

    Returns:
        dict: Dictionary containing the alignment matrix, coverage, repeat overlap, insert size outliers and discordant orientation.
    """    
    
    ### Initialize result dictionary
    result = dict()

    ### Load BAM file
    bam = pysam.AlignmentFile(bam_name, 'rb')
    
    ### Compute alignment matrix
    aln_matrix_left, aux_dict_left = compute_aln_matrix(bam, chrom, start - window, start + window, collapse_ins=collapse_ins)
    aln_matrix_right, aux_dict_right = compute_aln_matrix(bam, chrom, end - window, end + window, collapse_ins=collapse_ins)
    aln_matrix_left, aln_matrix_right = pad_alignment_matrices(aln_matrix_left, aln_matrix_right)
    result['aln_matrix_left'] = aln_matrix_left
    result['aux_dict_left'] = aux_dict_left 
    result['aln_matrix_right'] = aln_matrix_right
    result['aux_dict_right'] = aux_dict_right 
    
    ### Concatenate alignment matrices
    aln_matrix_middle = np.ones((aln_matrix_left.shape[0], 50)) * -1
    aln_matrix = np.concatenate((aln_matrix_left, aln_matrix_middle, aln_matrix_right), axis=1)
    result['aln_matrix'] = aln_matrix
    
    ### Do not compute coverage if SV too big
    if end - start < 5000000:
        compute_cov = True
    else:
        compute_cov = False
    result['compute_cov'] = compute_cov
        
    ### Compute coverage
    if compute_cov:
        cov_padding = max(padding, int((end - start) * 0.2))
        cov, cov_minq = compute_cov_df(bam_name, chrom, max(1, start - cov_padding), end + cov_padding)
        
        result['cov_padding'] = cov_padding
        result['cov'] = cov
        result['cov_minq'] = cov_minq
        if end - start > 1000 and end - start < 100000:
            cov_minq_smoothed = savgol_filter(cov_minq[2], int((end - start) * 0.05), 3)
            result['cov_minq_smoothed'] = cov_minq_smoothed
            
        ### Compute insert size outliers and discordant orientation
        isize_orient_dict = compute_isize_orientation_dict(bam_name, chrom, max(1, start - cov_padding), end + cov_padding)
        result['isize_orient_dict'] = isize_orient_dict
        
        
    ### Compute repeat overlap
    rep_df_sample = compute_rep_df(rep_df, chrom, start, end, padding=padding)
    result['rep_df'] = rep_df_sample
    
    return result


def plot_cov(start: int, end: int, data: dict, ax_cov_ill: plt.Axes, padding: int, baseline_cov: float=None):
    """ Plots coverage for a given region.

    Args:
        start (int): Start position
        end (int): End position
        data (dict): Sequencing data, result of gather_data_ill
        ax_cov_ill (plt.Axes): Axes to plot on
        padding (int): Padding around the SV to compute coverage.
        baseline_cov (float, optional): Chromosomal average coverage. Defaults to None.
    """    
    
    ### Extract data
    cov = data['cov']
    cov_minq = data['cov_minq']
    
    ### Plot 
    ax_cov_ill.plot(cov_minq[1], cov_minq[2], color='#df624c', fillstyle='bottom')
    ax_cov_ill.plot(cov[1], cov[2], color='grey', fillstyle='bottom')
    if end - start > 1000 and end - start < 100000:
        cov_minq_smoothed = data['cov_minq_smoothed']
        ax_cov_ill.plot(cov_minq[1], cov_minq_smoothed, color='black')
    if baseline_cov is not None:
        ax_cov_ill.axhline(y=baseline_cov, color='#df624c', linewidth=1, linestyle='--')
    ax_cov_ill.fill_between(cov[1], cov[2], color="grey", alpha=0.2)
    ax_cov_ill.axvline(x=padding, color='black', linewidth=1, linestyle='--')
    ax_cov_ill.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    ax_cov_ill.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax_cov_ill.set_xticks([])
    yticks = ax_cov_ill.get_yticks()
    yticks_filtered = yticks[yticks >= 0]
    ax_cov_ill.set_yticks(yticks_filtered)
    ax_cov_ill.set_ylim(bottom=-70, top=min(yticks[-1], 4 * baseline_cov))
    ax_cov_ill.text(0, 1.05, 'Coverage', transform=ax_cov_ill.transAxes, ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))
    
    
def plot_rep(data: dict, ax_cov_ill: plt.Axes, rep_y_pos_map: dict):
    """ Plots repeat overlap.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_cov_ill (plt.Axes): Axes to plot on
        rep_y_pos_map (dict): Dictionary mapping repeat types to y positions and colors.
    """    
    
    ### Extract data
    rep_df = data['rep_df']
    
    ### Plot
    for i in range(len(rep_df)):
        try:
            ax_cov_ill.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
        except KeyError: # repeat type not in rep_y_pos_map
            continue
    
    ### Add repeat labels
    for key in rep_y_pos_map.keys():
        ax_cov_ill.text(10, rep_y_pos_map[key][0]-2, key.replace('_', ' '), fontsize=7, horizontalalignment='left', verticalalignment='center', color='black', weight='bold')
        

def plot_isize(data: dict, ax_isize_ill: plt.Axes, padding: int):
    """ Plots insert size outliers.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_isize_ill (plt.Axes): Axes to plot on
        padding (int): Padding around the SV to compute coverage.
    """    
    
    ### Extract data
    isize_orient_dict = data['isize_orient_dict']
    cov = data['cov']
    x_range = np.linspace(0, cov.iloc[-1, 1], 1000)
    
    ### Helper function for histogram plotting with line plots
    def plot_hist_line(data_points, color, bins=50):
        if len(data_points) > 0:
            hist, bin_edges = np.histogram(data_points, bins=bins, range=(0, cov.iloc[-1, 1]), density=False)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax_isize_ill.plot(bin_centers, hist, color=color)
        else:
            ax_isize_ill.plot(x_range, np.zeros(1000), color=color)
        
        return len(data_points)
    
    ### Plot histogram for insert size outliers and get max value
    num_outliers = plot_hist_line(isize_orient_dict.get('exceed_max', []), color='black')
        
    ### Set plot limits and labels
    ax_isize_ill.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax_isize_ill.set_xticks([])
    ax_isize_ill.axvline(x=padding, color='black', linewidth=1, linestyle='--')
    ax_isize_ill.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    ax_isize_ill.text(0, 1.05, 'Insert Size Outliers', transform=ax_isize_ill.transAxes, ha='left',
                      bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))

    ### Configure right y-axis
    ax_isize_ill.yaxis.set_label_position("right")
    ax_isize_ill.yaxis.tick_right()
    yticks = ax_isize_ill.get_yticks()
    if len(yticks) > 0:
        ax_isize_ill.set_yticks([yticks[-1]])
    if num_outliers == 0:
        ax_isize_ill.set_yticks([])


def plot_orient(data: dict, ax_orient_ill: plt.Axes, padding: int):
    """ Plots discordant orientation.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_orient_ill (plt.Axes): Axes to plot on
        padding (int): Padding around the SV to compute coverage.
    """    
    
    ### Extract data
    isize_orient_dict = data['isize_orient_dict']
    cov = data['cov']
    x_range = np.linspace(0, cov.iloc[-1, 1], 1000)
    
    ### Helper function for histogram plotting
    def plot_hist(data_points, color, linestyle='-', bins=55):
        if len(data_points) > 0:
            hist, bin_edges = np.histogram(data_points, bins=bins, range=(0, cov.iloc[-1, 1]), density=False)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ax_orient_ill.plot(bin_centers, hist, color=color, linestyle=linestyle)
        else:
            ax_orient_ill.plot(x_range, np.zeros(1000), color=color)
            
        return len(data_points)
    
    ### Plot each orientation with scaling by number of data points
    num_rr = plot_hist(isize_orient_dict.get('rr', []), color='sandybrown')
    num_ff = plot_hist(isize_orient_dict.get('ff', []), color='cadetblue')
    num_rf = plot_hist(isize_orient_dict.get('rf', []), color='midnightblue')
    num_tx = plot_hist(isize_orient_dict.get('tx', []), color='grey', linestyle='--')
    
    ### Set plot limits and labels
    ax_orient_ill.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax_orient_ill.axvline(x=padding, color='black', linewidth=1, linestyle='--')
    ax_orient_ill.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    ax_orient_ill.text(0, 1.05, 'Discordant Read Pairs', transform=ax_orient_ill.transAxes, ha='left',
                       bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))
    
    ### Configure right y-axis
    ax_orient_ill.yaxis.set_label_position("right")
    ax_orient_ill.yaxis.tick_right()
    yticks = ax_orient_ill.get_yticks()
    if len(yticks) > 0:
        ax_orient_ill.set_yticks([yticks[-1]])
    if max(num_rr, num_ff, num_rf, num_tx) == 0:
        ax_orient_ill.set_yticks([])
    
    
def plot_cigar(data: dict, ax_cig_ill: plt.Axes, colors: list, window: int, tech: str):
    """ Plots CIGAR string information around breakpoints.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_cig_ill (plt.Axes): Axes to plot on
        colors (list): List of colors for the alignment matrix
        window (int): Window around each breakpoint to collect CIGAR string information for.
    """    
    
    ### Extract data
    aln_matrix = data['aln_matrix']
    aux_dict_left = data['aux_dict_left']
    aln_matrix_left = data['aln_matrix_left']
    aux_dict_right = data['aux_dict_right']
    aln_matrix_right = data['aln_matrix_right']
    
    ### Plot
    im = ax_cig_ill.imshow(aln_matrix, cmap=ListedColormap(colors), vmin=-1, vmax=8)

    ### Left Breakpoint 
    ax_cig_ill.axvline(x=window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_left, aln_matrix_left, ax_cig_ill)
    if tech == 'ill':
        add_disco_overlay(aux_dict_left, aln_matrix_left, ax_cig_ill, 'rr')
        add_disco_overlay(aux_dict_left, aln_matrix_left, ax_cig_ill, 'ff')
        add_disco_overlay(aux_dict_left, aln_matrix_left, ax_cig_ill, 'rf')
        add_disco_overlay(aux_dict_left, aln_matrix_left, ax_cig_ill, 'tx')
    add_splitread_overlay(aux_dict_left, aln_matrix_left, ax_cig_ill)

    ### Right Breakpoint
    ax_cig_ill.axvline(x=window+2.5*window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_right, aln_matrix_right, ax_cig_ill, offset=2.5*window)
    if tech == 'ill':
        add_disco_overlay(aux_dict_right, aln_matrix_right, ax_cig_ill, 'rr', offset=2.5*window)
        add_disco_overlay(aux_dict_right, aln_matrix_right, ax_cig_ill, 'ff', offset=2.5*window)
        add_disco_overlay(aux_dict_right, aln_matrix_right, ax_cig_ill, 'rf', offset=2.5*window)
        add_disco_overlay(aux_dict_right, aln_matrix_right, ax_cig_ill, 'tx', offset=2.5*window)
    add_splitread_overlay(aux_dict_right, aln_matrix_right, ax_cig_ill, offset=2.5*window)

    ### Read Connections
    df_aux_left = pd.DataFrame(aux_dict_left['name'], columns=['name']).reset_index()
    df_aux_right = pd.DataFrame(aux_dict_right['name'], columns=['name']).reset_index()
    df_aux_merge = df_aux_left.merge(df_aux_right, on='name', how='left', suffixes=('_left', '_right'))
    df_aux_merge = df_aux_merge[df_aux_merge['index_right'].notna()].reset_index(drop=True)
    df_aux_merge['index_right'] = df_aux_merge['index_right'].astype(int)
    df_aux_merge['split_left'] = 0
    df_aux_merge.loc[df_aux_merge['index_left'].isin(aux_dict_left['split_idx']), 'split_left'] = 1
    df_aux_merge['split_right'] = 0
    df_aux_merge.loc[df_aux_merge['index_right'].isin(aux_dict_right['split_idx']), 'split_right'] = 1

    for i in range(len(df_aux_merge)):
        if df_aux_merge.loc[i, 'split_left'] == 1 and df_aux_merge.loc[i, 'split_right'] == 1:
            ax_cig_ill.plot([window*2, (window*2)+49.5], [df_aux_merge.loc[i, 'index_left'], df_aux_merge.loc[i, 'index_right']], color='black', linewidth=0.5, linestyle='dotted')
        else:
            ax_cig_ill.plot([window*2, (window*2)+49.5], [df_aux_merge.loc[i, 'index_left'], df_aux_merge.loc[i, 'index_right']], color='#df624c', linewidth=0.5, linestyle='dotted')

    ax_cig_ill.axvline(x=(window*2)-0.5, color='lightgrey', linewidth=1, linestyle='--')
    ax_cig_ill.axvline(x=(window*2)+49.5, color='lightgrey', linewidth=1, linestyle='--')
    ax_cig_ill.set_xticks([0, window/2, window, window*1.5, 2*window, 2*window+50, 2*window+50+window/2, 2*window+50+window, 2*window+50+window*1.5, 2*window+50+2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window)), str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window))])
    ax_cig_ill.set_yticks([])
    ax_cig_ill.text(0.175, 1.05, 'Reads Left Breakpoint', transform=ax_cig_ill.transAxes, ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))
    ax_cig_ill.text(0.825, 1.05, 'Reads Right Breakpoint', transform=ax_cig_ill.transAxes, ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))
    ax_cig_ill.text(0.535, 1.05, 'Read Connections', transform=ax_cig_ill.transAxes, ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))
    
    
def plot_breakpoints_ill(samples: dict, rep_df: pd.DataFrame, sv_type: str, chrom: str, start: int, end: int, padding: int=1500, window: int=100, collapse_ins: bool=True, outfile: str=None):
    """ Visualizes alignment information around a structural variant for one or multiple samples.

    Args:
        samples (dict): Dictionary containing sample information. Keys are sample names and values are dictionaries containing family status, disease status, BAM filename and baseline coverage.
        rep_df (pd.DataFrame): Repeat dataframe
        sv_type (str): Type of the structural variant
        chrom (str): Chromosome
        start (int): Start position
        end (int): End position
        padding (int, optional): Padding around the SV to compute coverage.. Defaults to 1500.
        window (int, optional): Window around each breakpoint to collect CIGAR string information for.. Defaults to 100.
        collapse_ins (bool, optional): Collapse insertions into one base.. Defaults to True.
        outfile (str, optional): Output filename. Defaults to None.
    """    
    
    ### Set up figure
    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c', 'lightgrey', 'lightgrey']
    num_samples = len(samples)
    num_rows = 0
    height_ratios = []
    for sample in samples:
        technology = samples[sample]['technology']
        if technology == 'ill':
            num_rows += 5
            height_ratios.extend([1,3,0.5,0.5,7])
        elif technology == 'pb':
            num_rows += 3
            height_ratios.extend([1,3,7])
        
    fig = plt.figure(figsize=(25, 11 * num_samples))
    gs = gridspec.GridSpec(num_rows, 4, height_ratios=height_ratios, hspace=0.2, wspace=0.5)
    plt.rcParams['hatch.linewidth'] = 0.5
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    ### Set up parameters
    padding = max(padding, int((end - start) * 0.2))
    i = 0
    plotted_main_title = False
    for sample in samples:
        ### Get sample info
        name = sample
        family_status = samples[sample]['family_status']
        disease_status = samples[sample]['disease_status']
        technology = samples[sample]['technology']
        bam_name = samples[sample]['bam_name']
        baseline_cov = samples[sample]['baseline_cov']
        
        ### Extract Sequencing Data
        data = gather_data_ill(bam_name, chrom, start, end, window, padding, collapse_ins, rep_df)  
        
        ### Define sample axes
        if technology == 'ill':
            ax_title = plt.subplot(gs[i, 0:4])
            ax_cov_ill = plt.subplot(gs[i + 1, 0:4])
            ax_isize_ill = plt.subplot(gs[i + 2, 0:4])
            ax_orient_ill = plt.subplot(gs[i + 3, 0:4])
            ax_cig_ill = plt.subplot(gs[i + 4, 0:4])
            axs = [ax_title, ax_cov_ill, ax_isize_ill, ax_orient_ill, ax_cig_ill]
            for ax in axs:
                ax.grid(False)
                ax.set_facecolor('white')
            
            ### Create plots for current sample
            if data['compute_cov']:
                plot_cov(start, end, data, ax_cov_ill, padding, baseline_cov)
                plot_rep(data, ax_cov_ill, rep_y_pos_map)
                plot_isize(data, ax_isize_ill, padding)
                plot_orient(data, ax_orient_ill, padding)
            plot_cigar(data, ax_cig_ill, colors, window, 'ill')
            
            ### Update index
            i += 5
            
        elif technology == 'pb':
            ax_title = plt.subplot(gs[i, 0:4])
            ax_cov_pb = plt.subplot(gs[i + 1, 0:4])
            ax_cig_pb = plt.subplot(gs[i + 2, 0:4])
            axs = [ax_title, ax_cov_pb, ax_cig_pb]
            for ax in axs:
                ax.grid(False)
                ax.set_facecolor('white')
            
            ### Create plots for current sample
            if data['compute_cov']:
                plot_cov(start, end, data, ax_cov_pb, padding, baseline_cov)
                plot_rep(data, ax_cov_pb, rep_y_pos_map)
            plot_cigar(data, ax_cig_pb, colors, window, 'pb_clr')
            
            ### Update index
            i += 3
        
        ### Set title
        if not plotted_main_title:
            sv_len = end - start
            ax_title.text(0.5, 0.5, sv_type + ' ' + chrom + ':' + add_comma_to_pos(start) + '-' + add_comma_to_pos(end) + ' (' + add_comma_to_pos(sv_len) + ' bp)', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
            plotted_main_title = True
        ax_title.text(0.5, 0, name + ' (' + family_status.capitalize() + ', ' + disease_status.capitalize() + ')', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
        ax_title.axis('off')
    
    ### Save figure
    if outfile != None:
        try:
            plt.savefig(outfile, dpi=96, bbox_inches='tight')
            plt.close()
        except OverflowError:
            plt.clf()
            plt.text(0.5, 0.5, 'Plotting not possible', ha='center', va='center')
            plt.savefig(outfile, dpi=96, bbox_inches='tight')
    else:
        plt.show()

    
def plot_breakpoints_ill_pb(bam_filename_ill, bam_filename_pb, rep_df, chrom, leftbp, rightbp, padding=500, window=50, collapse_ins=True, title=None, outfile=None, df_svs_ill=None, df_svs_pb=None, ill_baseline_cov=None, pb_baseline_cov=None):
    """ Plots coverage and alignments around breakpoints. """
    ### Load BAM file
    bam1 = pysam.AlignmentFile(bam_filename_ill, 'rb')
    bam2 = pysam.AlignmentFile(bam_filename_pb, 'rb')

    ### Compute alignment matrix
    aln_matrix_left1, aux_dict_left1 = compute_aln_matrix(bam1, chrom, leftbp - window, leftbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_right1, aux_dict_right1 = compute_aln_matrix(bam1, chrom, rightbp - window, rightbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_left1, aln_matrix_right1 = pad_alignment_matrices(aln_matrix_left1, aln_matrix_right1)

    aln_matrix_left2, aux_dict_left2 = compute_aln_matrix(bam2, chrom, leftbp - window, leftbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_right2, aux_dict_right2 = compute_aln_matrix(bam2, chrom, rightbp - window, rightbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_left2, aln_matrix_right2 = pad_alignment_matrices(aln_matrix_left2, aln_matrix_right2)

    ### Concatenate alignment matrices
    aln_matrix_middle1 = np.ones((aln_matrix_left1.shape[0], 50)) * -1
    aln_matrix1 = np.concatenate((aln_matrix_left1, aln_matrix_middle1, aln_matrix_right1), axis=1)

    aln_matrix_middle2 = np.ones((aln_matrix_left2.shape[0], 50)) * -1
    aln_matrix2 = np.concatenate((aln_matrix_left2, aln_matrix_middle2, aln_matrix_right2), axis=1)

    ### Do not compute coverage if SV too big
    if rightbp - leftbp < 2000000:
        compute_cov = True
    else:
        compute_cov = False

    ### Compute coverage
    if compute_cov:
        cov_padding = max(padding, int((rightbp - leftbp) * 0.2))
        cov1, cov_minq1 = compute_cov_df(bam_filename_ill, chrom, max(1, leftbp - cov_padding), rightbp + cov_padding)
        cov2, cov_minq2 = compute_cov_df(bam_filename_pb, chrom, max(1, leftbp - cov_padding), rightbp + cov_padding)
        if rightbp - leftbp > 1000 and rightbp - leftbp < 100000:
            cov_minq1_smoothed = savgol_filter(cov_minq1[2], int((rightbp - leftbp) * 0.05), 3)
            cov_minq2_smoothed = savgol_filter(cov_minq2[2], int((rightbp - leftbp) * 0.05), 3)

    ### Compute repeat overlap
    rep_df = compute_rep_df(rep_df, chrom, leftbp, rightbp)

    ### Plot options
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c', 'lightgrey', 'lightgrey']
    fig = plt.figure(figsize=(25,22))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(6, 4, height_ratios=[3,1,6,3,1,6], hspace=0.2, wspace=0.5)
    ax_cov_ill = plt.subplot(gs[0, 0:4])
    ax_svs_ill = plt.subplot(gs[1, 0:4])
    ax_cig_ill = plt.subplot(gs[2, 0:4])
    ax_cov_pb = plt.subplot(gs[3, 0:4])
    ax_svs_pb = plt.subplot(gs[4, 0:4])
    ax_cig_pb = plt.subplot(gs[5, 0:4])
    
    axs = [ax_cov_ill, ax_svs_ill, ax_cig_ill, ax_cov_pb, ax_svs_pb, ax_cig_pb]
    
    for ax in axs:
        ax.grid(False)
        ax.set_facecolor('white')
        
    for ax in [ax_cig_ill, ax_cig_pb]:
        ax.axes.yaxis.set_visible(False)
        
    for ax in [ax_svs_ill, ax_svs_pb]:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    if compute_cov:
        ### Coverage 1
        ax_cov_ill.plot(cov_minq1[1], cov_minq1[2], color='#df624c', fillstyle='bottom')
        ax_cov_ill.plot(cov1[1], cov1[2], color='grey', fillstyle='bottom')
        if rightbp - leftbp > 1000 and rightbp - leftbp < 100000:
            ax_cov_ill.plot(cov_minq1[1], cov_minq1_smoothed, color='black')
        if ill_baseline_cov is not None:
            ax_cov_ill.axhline(y=ill_baseline_cov, color='#df624c', linewidth=1, linestyle='--')
        ax_cov_ill.fill_between(cov1[1], cov1[2], color="grey", alpha=0.2)
        ax_cov_ill.axvline(x=cov_padding, color='black', linewidth=1, linestyle='--')
        ax_cov_ill.axvline(x=cov1.iloc[-1, 1] - cov_padding, color='black', linewidth=1, linestyle='--')
        cov1_median = cov1[2].median()
        cov1_mad = (cov1[2] - cov1_median).abs().median()
        ax_cov_ill.set_ylim(bottom=-70)
        ax_cov_ill.set_xlim(left=0, right=cov1.iloc[-1, 1])
        
        ### Repeat track 1
        for i in range(len(rep_df)):
            try:
                ax_cov_ill.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
            except KeyError: # repeat type not in rep_y_pos_map
                continue
        
        # Add repeat labels
        for key in rep_y_pos_map.keys():
            ax_cov_ill.text(10, rep_y_pos_map[key][0]-2, key.replace('_', ' '), fontsize=7, horizontalalignment='left', verticalalignment='center', color='black', weight='bold')
        
        ### Coverage 2
        ax_cov_pb.plot(cov_minq2[1], cov_minq2[2], color='#df624c', fillstyle='bottom')
        ax_cov_pb.plot(cov2[1], cov2[2], color='grey', fillstyle='bottom')
        if rightbp - leftbp > 1000 and rightbp - leftbp < 100000:
            ax_cov_pb.plot(cov_minq2[1], cov_minq2_smoothed, color='black')
        if pb_baseline_cov is not None:
            ax_cov_pb.axhline(y=pb_baseline_cov, color='#df624c', linewidth=1, linestyle='--')
        ax_cov_pb.fill_between(cov2[1], cov2[2], color="grey", alpha=0.2)
        ax_cov_pb.axvline(x=cov_padding, color='black', linewidth=1, linestyle='--')
        ax_cov_pb.axvline(x=cov2.iloc[-1, 1] - cov_padding, color='black', linewidth=1, linestyle='--')
        cov2_median = cov2[2].median()
        cov2_mad = (cov2[2] - cov2_median).abs().median()
        ax_cov_pb.set_ylim(bottom=-70)
        ax_cov_pb.set_xlim(left=0, right=cov2.iloc[-1, 1])
        #ax_cov_pb.set_yticks([0, 20, 40, 60, 80], labels=['0', '20', '40', '60', ''])
        
        ### Repeat track 2
        for i in range(len(rep_df)):
            try:
                ax_cov_pb.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
            except KeyError: # repeat type not in rep_y_pos_map
                continue
        
        # Add repeat labels
        for key in rep_y_pos_map.keys():
            ax_cov_pb.text(10, rep_y_pos_map[key][0]-2, key.replace('_', ' '), fontsize=7, horizontalalignment='left', verticalalignment='center', color='black', weight='bold')
    
    plt.rcParams['hatch.linewidth'] = 0.5
    cmap = cm.inferno.reversed()
    
    ### SV Neighbourhood ILL
    if compute_cov:
        ax_svs_ill.set_xlim(left=0, right=rightbp - leftbp + 2 * cov_padding)
    else:
        ax_svs_ill.set_xlim(left=0, right=rightbp - leftbp + 2 * padding)
    if df_svs_ill is not None:
        if compute_cov:
            sv_neighbourhood_ill_df = get_variant_neighbourhood(df_svs_ill, chrom, leftbp, rightbp, padding=cov_padding)
        else:
            sv_neighbourhood_ill_df = get_variant_neighbourhood(df_svs_ill, chrom, leftbp, rightbp, padding=padding)
        for i in range(len(sv_neighbourhood_ill_df)):
            y_pos = i * -8
            sv_qual = sv_neighbourhood_ill_df.loc[i, 'dicast_qual']
            if not np.isnan(sv_qual):
                color = cmap(sv_qual)
            else:
                color = 'lightgrey'
            ax_svs_ill.hlines(y=y_pos, xmin=sv_neighbourhood_ill_df.loc[i, 'start'], xmax=sv_neighbourhood_ill_df.loc[i, 'end'], linewidth=4, color=color)
            ax_svs_ill.text(10, y_pos, sv_neighbourhood_ill_df.loc[i, 'caller'], fontsize=7, horizontalalignment='left', verticalalignment='center', color='black', weight='bold')
    

    ### Breakpoints ILL
    im = ax_cig_ill.imshow(aln_matrix1, cmap=ListedColormap(colors), vmin=-1, vmax=8)

    ### Left Breakpoint 1
    ax_cig_ill.axvline(x=window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_left1, aln_matrix_left1, ax_cig_ill)
    add_disco_overlay(aux_dict_left1, aln_matrix_left1, ax_cig_ill, 'rr')
    add_disco_overlay(aux_dict_left1, aln_matrix_left1, ax_cig_ill, 'ff')
    add_disco_overlay(aux_dict_left1, aln_matrix_left1, ax_cig_ill, 'rf')
    add_splitread_overlay(aux_dict_left1, aln_matrix_left1, ax_cig_ill)

    ### Right Breakpoint 1
    ax_cig_ill.axvline(x=window+2.5*window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_right1, aln_matrix_right1, ax_cig_ill, offset=2.5*window)
    add_disco_overlay(aux_dict_right1, aln_matrix_right1, ax_cig_ill, 'rr', offset=2.5*window)
    add_disco_overlay(aux_dict_right1, aln_matrix_right1, ax_cig_ill, 'ff', offset=2.5*window)
    add_disco_overlay(aux_dict_right1, aln_matrix_right1, ax_cig_ill, 'rf', offset=2.5*window)
    add_splitread_overlay(aux_dict_right1, aln_matrix_right1, ax_cig_ill, offset=2.5*window)

    ### Read Connections 1
    df_aux_left1 = pd.DataFrame(aux_dict_left1['name'], columns=['name']).reset_index()
    df_aux_right1 = pd.DataFrame(aux_dict_right1['name'], columns=['name']).reset_index()
    df_aux_merge1 = df_aux_left1.merge(df_aux_right1, on='name', how='left', suffixes=('_left', '_right'))
    df_aux_merge1 = df_aux_merge1[df_aux_merge1['index_right'].notna()].reset_index(drop=True)
    df_aux_merge1['index_right'] = df_aux_merge1['index_right'].astype(int)
    df_aux_merge1['split_left'] = 0
    df_aux_merge1.loc[df_aux_merge1['index_left'].isin(aux_dict_left1['split_idx']), 'split_left'] = 1
    df_aux_merge1['split_right'] = 0
    df_aux_merge1.loc[df_aux_merge1['index_right'].isin(aux_dict_right1['split_idx']), 'split_right'] = 1

    for i in range(len(df_aux_merge1)):
        if df_aux_merge1.loc[i, 'split_left'] == 1 and df_aux_merge1.loc[i, 'split_right'] == 1:
            ax_cig_ill.plot([window*2, (window*2)+49.5], [df_aux_merge1.loc[i, 'index_left'], df_aux_merge1.loc[i, 'index_right']], color='black', linewidth=0.5, linestyle='dotted')
        else:
            ax_cig_ill.plot([window*2, (window*2)+49.5], [df_aux_merge1.loc[i, 'index_left'], df_aux_merge1.loc[i, 'index_right']], color='#df624c', linewidth=0.5, linestyle='dotted')

    ax_cig_ill.axvline(x=(window*2)-0.5, color='lightgrey', linewidth=1, linestyle='--')
    ax_cig_ill.axvline(x=(window*2)+49.5, color='lightgrey', linewidth=1, linestyle='--')
    ax_cig_ill.set_xticks([0, window/2, window, window*1.5, 2*window, 2*window+50, 2*window+50+window/2, 2*window+50+window, 2*window+50+window*1.5, 2*window+50+2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window)), str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window))])

    ### Breakpoints PB
    im = ax_cig_pb.imshow(aln_matrix2, cmap=ListedColormap(colors), vmin=-1, vmax=8)

    ### Left Breakpoint 2
    ax_cig_pb.axvline(x=window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_left2, aln_matrix_left2, ax_cig_pb)
    add_splitread_overlay(aux_dict_left2, aln_matrix_left2, ax_cig_pb)

    ### Right Breakpoint 2
    ax_cig_pb.axvline(x=window+2.5*window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_right2, aln_matrix_right2, ax_cig_pb, offset=2.5*window)
    add_splitread_overlay(aux_dict_right2, aln_matrix_right2, ax_cig_pb, offset=2.5*window)

    ### Read Connections 2
    df_aux_left2 = pd.DataFrame(aux_dict_left2['name'], columns=['name']).reset_index()
    df_aux_right2 = pd.DataFrame(aux_dict_right2['name'], columns=['name']).reset_index()
    df_aux_merge2 = df_aux_left2.merge(df_aux_right2, on='name', how='left', suffixes=('_left', '_right'))
    df_aux_merge2 = df_aux_merge2[df_aux_merge2['index_right'].notna()].reset_index(drop=True)
    df_aux_merge2['index_right'] = df_aux_merge2['index_right'].astype(int)

    for i in range(len(df_aux_merge2)):
        ax_cig_pb.plot([window*2, (window*2)+49.5], [df_aux_merge2.loc[i, 'index_left'], df_aux_merge2.loc[i, 'index_right']], color='black', linewidth=0.5, linestyle='dotted')

    ax_cig_pb.axvline(x=(window*2)-0.5, color='lightgrey', linewidth=1, linestyle='--')
    ax_cig_pb.axvline(x=(window*2)+49.5, color='lightgrey', linewidth=1, linestyle='--')
    ax_cig_pb.set_xticks([0, window/2, window, window*1.5, 2*window, 2*window+50, 2*window+50+window/2, 2*window+50+window, 2*window+50+window*1.5, 2*window+50+2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window)), str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window))])

    ### Output
    if title != None:
        ax_cov_ill.set_title(title)
    if outfile != None:
        try:
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close()
        except OverflowError:
            plt.clf()
            plt.text(0.5, 0.5, 'Plotting not possible', ha='center', va='center')
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def acc_dot(aln_matrix, ax, labels, color=None):
    """ Calculates x and y positions for dotplot. """

    if len(labels) > 0:
        # Color dict for haplotag
        color_dict_hp = {'-1' : 'black', '1': 'blue', '2': 'red'}
        

    # Plot dotplot
    for i in range(len(aln_matrix)):
        x_pos = 0
        y_pos = 0
        all_x_pos = []
        all_y_pos = []
        for j in range(len(aln_matrix[i])):
            if aln_matrix[i][j] >= 0:
                all_x_pos.append(x_pos)
                all_y_pos.append(y_pos)
            if aln_matrix[i][j] <= 0:
                x_pos += 1
                y_pos += 1
            elif aln_matrix[i][j] == 1:
                y_pos += 1
            elif aln_matrix[i][j] == 2:
                x_pos += 1
            elif aln_matrix[i][j] > 3:
                x_pos += 1
        if len(labels) > 0:
            ax.plot(all_x_pos, all_y_pos, alpha=1/len(aln_matrix)+0.3, linewidth=3, color=color_dict_hp[labels[i]] if len(labels) > 0 else 'red')
        
        elif color is not None:
            ax.plot(all_x_pos, all_y_pos, alpha=1/len(aln_matrix)+0.3, linewidth=3, color=color)
            
        else:
            ax.plot(all_x_pos, all_y_pos, alpha=1/len(aln_matrix)+0.3, linewidth=3, color='red')


def plot_dotplots(bam_filename, chrom, left_bp, right_bp, padding=100, color_by='', outfile=''):
    """ Computes alignment matrix and plots dotplot. """
    bam = pysam.AlignmentFile(bam_filename, 'rb')
    
    # Compute alignment matrix
    size = right_bp - left_bp + padding * 2
    aln_matrix, aux_dict = compute_aln_matrix(bam, chrom, left_bp-padding, right_bp+padding, collapse_ins=False, size=size)

    # Plot dotplot
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    fig.patch.set_facecolor('white')

    # Coloring options
    if color_by == 'HP':
        labels = [str(x) for x in aux_dict['haplotag_idx']]
    else:
        labels = []

    acc_dot(aln_matrix, ax, labels)
    ax.vlines(x=padding, ymin=-1, ymax=size, color='black', linewidth=1, alpha=0.5, linestyles='dashed')
    ax.vlines(x=size-padding, ymin=-1, ymax=size, color='black', linewidth=1, alpha=0.5, linestyles='dashed')

    ax.set_ylim(-1, len(aln_matrix[0]))
    ax.set_xlim(-1, len(aln_matrix[0]))
    ax.set_ylabel('Alignment Position')
    ax.set_xlabel('Reference Position')
    plt.title(chrom + ':' + str(left_bp) + '-' + str(right_bp))
    if outfile == '':
        plt.show()
    else:
        plt.savefig(outfile)
        
        
def plot_dotplot_multi_sample(bam_filename_1, bam_filename_2, sample1, sample2, chrom, left_bp, right_bp, padding=100, outfile=''):
    
    bam1 = pysam.AlignmentFile(bam_filename_1, 'rb')
    bam2 = pysam.AlignmentFile(bam_filename_2, 'rb')
    
    plt.rcParams['font.size'] = 18  # Adjusts the base/default font size
    plt.rcParams['axes.labelsize'] = 18  # For x and y labels
    plt.rcParams['axes.titlesize'] = 18  # For the title
    plt.rcParams['legend.fontsize'] = 26  # For legend
    
    # Compute alignment matrix
    size = right_bp - left_bp + padding * 2
    aln_matrix1, aux_dict1 = compute_aln_matrix(bam1, chrom, left_bp-padding, right_bp+padding, collapse_ins=False, size=size)
    aln_matrix2, aux_dict2 = compute_aln_matrix(bam2, chrom, left_bp-padding, right_bp+padding, collapse_ins=False, size=size)
    
    # Plot dotplot
    fig, axs = plt.subplots(1, 1, figsize=(12,12))
    #fig.patch.set_facecolor('white')
    
    # make background white
    #axs.set_facecolor('white')
    
    acc_dot(aln_matrix1, axs, [], color='blue')
    acc_dot(aln_matrix2, axs, [], color='red')
    axs.vlines(x=padding, ymin=-1, ymax=size, color='black', linewidth=3, alpha=0.5, linestyles='dashed')
    axs.vlines(x=size-padding, ymin=-1, ymax=size, color='black', linewidth=3, alpha=0.5, linestyles='dashed')
    
    axs.set_ylim(-1, len(aln_matrix1[0]))
    axs.set_xlim(-1, len(aln_matrix1[0]))
    axs.set_ylabel('Alignment Position')
    axs.set_xlabel('Reference Position')
    plt.title(chrom + ':' + str(left_bp) + '-' + str(right_bp))
    
    # Add legend
    blue_line = mlines.Line2D([], [], color='blue', markersize=20, label=sample1)
    red_line = mlines.Line2D([], [], color='red', markersize=20, label=sample2)
    axs.legend(handles=[blue_line, red_line], fontsize=20)  # Increase font size here
    
    # make legend background white
    axs.get_legend().get_frame().set_facecolor('white') 
    
    # remove legend border
    axs.get_legend().get_frame().set_linewidth(0.0)
    
    # make legend lines thicker
    for line in axs.get_legend().get_lines():
        line.set_linewidth(3)
        
    # make x and y axis visible
    axs.spines['top'].set_visible(True)
    axs.spines['right'].set_visible(True)
    axs.spines['bottom'].set_visible(True)
    axs.spines['left'].set_visible(True)
    
    # make x and y axis black
    axs.spines['top'].set_color('black')
    axs.spines['right'].set_color('black')
    axs.spines['bottom'].set_color('black')
    axs.spines['left'].set_color('black')
    
    
    
    if outfile == '':
        plt.show()
    else:
        plt.savefig(outfile)
