import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from matplotlib import cm
import numpy as np
import pysam
import random 
from scipy.signal import savgol_filter

from cuban_lib.utils import compute_aln_matrix, pad_alignment_matrices, compute_cov_df, compute_rep_df, get_variant_neighbourhood

mpl.rcParams['agg.path.chunksize'] = 1000000
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


rep_y_pos_map = {'LINE' : (-6, '#6ac0b7'),
                 'SINE' : (-14, '#b7954b'),
                 'LTR' : (-22, '#f0b6a0'),
                 'DNA' : (-30, '#5066a2'),
                 'Simple_repeat' : (-38, '#504669'),
                 'Satellite' : (-46, '#df624c'),
                 'Low_complexity' : (-54, '#61856b'),
                 'Retroposon' : (-62, '#2f7155')}


def add_splitread_overlay(aux_dict, aln_matrix_left1, ax, offset=0):

    for idx in aux_dict['split_idx']:
        try:
            start = np.where(aln_matrix_left1[idx] >= 4)[0][0]
            end = np.where(aln_matrix_left1[idx] >= 4)[0][-1]
            ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx - 0.5),end-start+1, 0.9, hatch='||', fill=False, snap=False, linewidth=0.5, edgecolor='black'))
        except IndexError:
            pass


def add_disco_overlay(aux_dict, aln_matrix, ax, orient, offset=0):
    
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
        except IndexError:
            pass


def add_mapq_overlay(aux_dict, aln_matrix, ax, offset=0):

    for idx in aux_dict['low_mapq_idx']:
        try:
            start = np.where(aln_matrix[idx] >= 0)[0][0]
            end = np.where(aln_matrix[idx] >= 0)[0][-1]
            ax.add_patch(mplpatches.Rectangle((start-0.5 + offset, idx-0.5),end-start+1,0.9,fill=True, snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass


def plot_breakpoints_ill(bam_filename, rep_filename, chrom, leftbp, rightbp, padding=500, window=50, collapse_ins=True, title=None, outfile=None):
    """ Plots coverage and alignments around breakpoints. """

    ### Load BAM file
    bam = pysam.AlignmentFile(bam_filename, 'rb')

    ### Compute alignment matrix
    aln_matrix_left, aux_dict_left = compute_aln_matrix(bam, chrom, leftbp - window, leftbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_right, aux_dict_right = compute_aln_matrix(bam, chrom, rightbp - window, rightbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_left, aln_matrix_right = pad_alignment_matrices(aln_matrix_left, aln_matrix_right)

    ### Concatenate alignment matrices
    aln_matrix_middle = np.ones((aln_matrix_left.shape[0], 50)) * -1
    aln_matrix = np.concatenate((aln_matrix_left, aln_matrix_middle, aln_matrix_right), axis=1)

    ### Do not compute coverage if SV too big
    if rightbp - leftbp < 100000:
        compute_cov = True
    else:
        compute_cov = False

    if compute_cov:
        ### Compute coverage
        cov, cov_minq = compute_cov_df(bam_filename, chrom, leftbp - padding, rightbp + padding)

        ### Compute repeat overlap
        rep_df = compute_rep_df(rep_filename, chrom, leftbp - 5000, rightbp + 5000)

    ### Plot options
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c', 'lightgrey', 'lightgrey']
    fig = plt.figure(figsize=(22,10))

    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(2, 4, height_ratios=[1,3], hspace=0.0, wspace=0.5)
    ax1 = plt.subplot(gs[0, 0:4])
    ax2 = plt.subplot(gs[1, 0:4])
    axs = [ax1, ax2]
    for ax in axs:
        ax.grid(False)
        ax.set_facecolor('white')
    ax2.axes.yaxis.set_visible(False)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    ### Coverage
    if compute_cov:
        ax1.plot(cov_minq[1], cov_minq[2], color='#df624c', fillstyle='bottom')
        ax1.plot(cov[1], cov[2], color='grey', fillstyle='bottom')
        ax1.fill_between(cov[1], cov[2], color="grey", alpha=0.2)
        ax1.axvline(x=500, color='black', linewidth=1, linestyle='--')
        ax1.axvline(x=cov.iloc[-1, 1]-500, color='black', linewidth=1, linestyle='--')
        ax1.set_ylim(bottom=-66)
        ax1.set_xlim(left=0, right=cov.iloc[-1, 1])
        ax1.set_yticks([0, 20, 40, 60], labels=['0', '20', '40', '60'])

        ### Repeat track
        for i in range(len(rep_df)):
            try:
                ax1.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
            except KeyError: # repeat type not in rep_y_pos_map
                continue

        # Add repeat labels
        for key in rep_y_pos_map.keys():
            ax1.text(10, rep_y_pos_map[key][0]-2, key.replace('_', ' '), fontsize=7, horizontalalignment='left', verticalalignment='center', color='black', weight='bold')

    ### Breakpoints
    im = ax2.imshow(aln_matrix, cmap=ListedColormap(colors), vmin=-1, vmax=5)

    ### Left Breakpoint
    ax2.axvline(x=window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_left, aln_matrix_left, ax2)
    add_disco_overlay(aux_dict_left, aln_matrix_left, ax2, 'rr')
    add_disco_overlay(aux_dict_left, aln_matrix_left, ax2, 'ff')
    add_splitread_overlay(aux_dict_left, aln_matrix_left, ax2)
    
    ### Right Breakpoint
    ax2.axvline(x=window+2.5*window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_dict_right, aln_matrix_right, ax2, offset=2.5*window)
    add_disco_overlay(aux_dict_right, aln_matrix_right, ax2, 'rr', offset=2.5*window)
    add_disco_overlay(aux_dict_right, aln_matrix_right, ax2, 'ff', offset=2.5*window)
    add_splitread_overlay(aux_dict_right, aln_matrix_right, ax2, offset=2.5*window)

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
            ax2.plot([window*2, (window*2)+49.5], [df_aux_merge.loc[i, 'index_left'], df_aux_merge.loc[i, 'index_right']], color='black', linewidth=0.5, linestyle='dotted')
        else:
            ax2.plot([window*2, (window*2)+49.5], [df_aux_merge.loc[i, 'index_left'], df_aux_merge.loc[i, 'index_right']], color='#df624c', linewidth=0.5, linestyle='dotted')

    ax2.axvline(x=(window*2)-0.5, color='lightgrey', linewidth=1, linestyle='--')
    ax2.axvline(x=(window*2)+49.5, color='lightgrey', linewidth=1, linestyle='--')
    ax2.set_xticks([0, window/2, window, window*1.5, 2*window, 2*window+50, 2*window+50+window/2, 2*window+50+window, 2*window+50+window*1.5, 2*window+50+2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window)), str(-int(window)), str(-int(window/2)), '0', str(int(window/2)), str(int(window))])

    ### Colorbar
    cbar = fig.colorbar(im, cmap=ListedColormap(colors[1:]), ax=[ax1, ax2], shrink=0.5, ticks=[0.3,1.1,2,2.8,3.7,4.5])
    labels = ['M', 'I', 'D', 'N', 'S', 'H']
    cbar.ax.set_yticklabels(labels)

    ### Output
    if title != None:
        ax1.set_title(title)
    if outfile != None:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
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


def acc_dot(aln_matrix, ax, labels):
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
        ax.plot(all_x_pos, all_y_pos, alpha=1/len(aln_matrix)+0.1, linewidth=1, color=color_dict_hp[labels[i]] if len(labels) > 0 else 'red')


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
