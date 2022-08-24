import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from matplotlib import cm
import numpy as np
import pysam
import random 

from lib.utils import compute_aln_matrix, pad_alignment_matrices, compute_cov_df, compute_rep_df

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


rep_y_pos_map = {'LINE' : (-6, '#6ac0b7'),
                 'SINE' : (-10, '#b7954b'),
                 'LTR' : (-14, '#f0b6a0'),
                 'DNA' : (-18, '#5066a2'),
                 'Simple_repeat' : (-22, '#504669'),
                 'Satellite' : (-26, '#df624c'),
                 'Low_complexity' : (-30, '#61856b'),
                 'Retroposon' : (-34, '#2f7155')}


def plot_breakpoints_single(bam_filename, rep_filename, chrom, leftbp, rightbp, padding=500, window=50, collapse_ins=True):
    """ Plots coverage and alignments around breakpoints. """
    ### Load BAM file
    bam = pysam.AlignmentFile(bam_filename, 'rb')

    ### Compute alignment matrix
    aln_matrix_left, aux_dict_left = compute_aln_matrix(bam, chrom, leftbp - window, leftbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_right, aux_dict_right = compute_aln_matrix(bam, chrom, rightbp - window, rightbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_left, aln_matrix_right = pad_alignment_matrices(aln_matrix_left, aln_matrix_right)

    ### Compute coverage
    cov, cov_minq = compute_cov_df(bam_filename, chrom, leftbp - padding, rightbp + padding)

    ### Compute repeat overlap
    rep_df = compute_rep_df(rep_filename, chrom, leftbp - 5000, rightbp + 5000)

    ### Plot options
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c']
    fig = plt.figure(figsize=(22,7))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 4, height_ratios=[1,3], hspace=0.0, wspace=0.5)
    ax1 = plt.subplot(gs[0, 0:4])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1, 2:])
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax2.axes.yaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    ax3.set_facecolor('white')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    ### Coverage
    ax1.plot(cov_minq[1], cov_minq[2], color='#df624c', fillstyle='bottom')
    ax1.plot(cov[1], cov[2], color='grey', fillstyle='bottom')
    ax1.fill_between(cov[1], cov[2], color="grey", alpha=0.2)
    ax1.axvline(x=500, color='black', linewidth=1, linestyle='--')
    ax1.axvline(x=cov.iloc[-1, 1]-500, color='black', linewidth=1, linestyle='--')
    ax1.set_ylim(bottom=-36)
    ax1.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax1.set_yticks([0, 20, 40, 60], labels=['0', '20', '40', '60'])

    ### Repeat track
    for i in range(len(rep_df)):
        try:
            ax1.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
        except KeyError: # repeat type not in rep_y_pos_map
            continue

    ### Left Breakpoint
    ax2.imshow(aln_matrix_left, cmap=ListedColormap(colors), vmin=-1, vmax=5)
    ax2.axvline(x=window, color='black', linewidth=1, linestyle='--')
    for idx in aux_dict_left['split_idx']:
        try:
            start = np.where(aln_matrix_left[idx] >= 4)[0][0]
            end = np.where(aln_matrix_left[idx] >= 4)[0][-1]
            ax2.add_patch(mplpatches.Rectangle((start, idx - 0.5),end-start,1,hatch='\\',fill=False,snap=False, linewidth=1, edgecolor='black'))
        except IndexError:
            pass
    for idx in aux_dict_left['low_mapq_idx']:
        try:
            start = np.where(aln_matrix_left[idx] >= 0)[0][0]
            end = np.where(aln_matrix_left[idx] >= 0)[0][-1]
            ax2.add_patch(mplpatches.Rectangle((start-0.5, idx - 0.5),end-start+1,1,hatch='\\',fill=True,snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
    #ax2.set_xticks([0, 25, 50, 75, 100], labels=['-50', '-25', '0', '25', '50'])

    ### Right Breakpoint
    im = ax3.imshow(aln_matrix_right, cmap=ListedColormap(colors), vmin=-1, vmax=5)
    ax3.axvline(x=window, color='black', linewidth=1, linestyle='--')
    for idx in aux_dict_right['split_idx']:
        try:
            start = np.where(aln_matrix_right[idx] >= 4)[0][0]
            end = np.where(aln_matrix_right[idx] >= 4)[0][-1]
            ax3.add_patch(mplpatches.Rectangle((start, idx - 0.5),end-start,1,hatch='\\',fill=False,snap=False, linewidth=1, edgecolor='black'))
        except IndexError:
            pass
    for idx in aux_dict_right['low_mapq_idx']:
        try:
            start = np.where(aln_matrix_right[idx] >= 0)[0][0]
            end = np.where(aln_matrix_right[idx] >= 0)[0][-1]
            ax3.add_patch(mplpatches.Rectangle((start-0.5, idx - 0.5),end-start+1,1,hatch='\\',fill=True,snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
    #ax3.set_xticks([0, 25, 50, 75, 100], labels=['-50', '-25', '0', '25', '50'])
    cbar = fig.colorbar(im, cmap=ListedColormap(colors[1:]), ax=[ax1, ax2, ax3], shrink=0.5, ticks=[0.3,1.1,2,2.8,3.7,4.5])
    labels = ['M', 'I', 'D', 'N', 'S', 'H']
    cbar.ax.set_yticklabels(labels)

    plt.show()


def plot_breakpoints_double(bam_filename1, bam_filename2, rep_filename, chrom, leftbp, rightbp, padding=500, window=50, collapse_ins=True, title=None, outfile=None):
    """ Plots coverage and alignments around breakpoints. """
    ### Load BAM file
    bam1 = pysam.AlignmentFile(bam_filename1, 'rb')
    bam2 = pysam.AlignmentFile(bam_filename2, 'rb')

    ### Compute alignment matrix
    aln_matrix_left1, aux_dict_left1 = compute_aln_matrix(bam1, chrom, leftbp - window, leftbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_right1, aux_dict_right1 = compute_aln_matrix(bam1, chrom, rightbp - window, rightbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_left1, aln_matrix_right1 = pad_alignment_matrices(aln_matrix_left1, aln_matrix_right1)

    aln_matrix_left2, aux_dict_left2 = compute_aln_matrix(bam2, chrom, leftbp - window, leftbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_right2, aux_dict_right2 = compute_aln_matrix(bam2, chrom, rightbp - window, rightbp + window, collapse_ins=collapse_ins, size=2*window)
    aln_matrix_left2, aln_matrix_right2 = pad_alignment_matrices(aln_matrix_left2, aln_matrix_right2)

    ### Compute coverage
    cov1, cov_minq1 = compute_cov_df(bam_filename1, chrom, leftbp - padding, rightbp + padding)
    cov2, cov_minq2 = compute_cov_df(bam_filename2, chrom, leftbp - padding, rightbp + padding)

    ### Compute repeat overlap
    rep_df = compute_rep_df(rep_filename, chrom, leftbp - 5000, rightbp + 5000)

    ### Plot options
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c']
    fig = plt.figure(figsize=(22,14))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(4, 4, height_ratios=[1,3,1,3], hspace=0.0, wspace=0.5)
    ax1 = plt.subplot(gs[0, 0:4])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1, 2:])
    ax4 = plt.subplot(gs[2, 0:4])
    ax5 = plt.subplot(gs[3, :2])
    ax6 = plt.subplot(gs[3, 2:])
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)
    ax5.grid(False)
    ax6.grid(False)
    ax2.axes.yaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)
    ax6.axes.yaxis.set_visible(False)
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    ax3.set_facecolor('white')
    ax4.set_facecolor('white')
    ax5.set_facecolor('white')
    ax6.set_facecolor('white')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    ### Coverage 1
    ax1.plot(cov_minq1[1], cov_minq1[2], color='#df624c', fillstyle='bottom')
    ax1.plot(cov1[1], cov1[2], color='grey', fillstyle='bottom')
    ax1.fill_between(cov1[1], cov1[2], color="grey", alpha=0.2)
    ax1.axvline(x=500, color='black', linewidth=1, linestyle='--')
    ax1.axvline(x=cov1.iloc[-1, 1]-500, color='black', linewidth=1, linestyle='--')
    ax1.set_ylim(bottom=-36)
    ax1.set_xlim(left=0, right=cov1.iloc[-1, 1])
    ax1.set_yticks([0, 20, 40, 60], labels=['0', '20', '40', '60'])

    ### Repeat track 1
    for i in range(len(rep_df)):
        try:
            ax1.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
        except KeyError: # repeat type not in rep_y_pos_map
            continue
    
    ### Coverage 2
    ax4.plot(cov_minq2[1], cov_minq2[2], color='#df624c', fillstyle='bottom')
    ax4.plot(cov2[1], cov2[2], color='grey', fillstyle='bottom')
    ax4.fill_between(cov2[1], cov2[2], color="grey", alpha=0.2)
    ax4.axvline(x=500, ymax=0.8, color='black', linewidth=1, linestyle='--')
    ax4.axvline(x=cov2.iloc[-1, 1]-500, ymax=0.8, color='black', linewidth=1, linestyle='--')
    ax4.set_ylim(bottom=-36)
    ax4.set_xlim(left=0, right=cov1.iloc[-1, 1])
    ax4.set_yticks([0, 20, 40, 60, 80], labels=['0', '20', '40', '60', ''])
    ax4.spines['top'].set_color('lightgrey')

    ### Repeat track 2
    for i in range(len(rep_df)):
        try:
            ax4.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0], xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
        except KeyError: # repeat type not in rep_y_pos_map
            continue

    ### Left Breakpoint 1
    ax2.imshow(aln_matrix_left1, cmap=ListedColormap(colors), vmin=-1, vmax=5)
    ax2.axvline(x=window, color='black', linewidth=1, linestyle='--')
    for idx in aux_dict_left1['split_idx']:
        try:
            start = np.where(aln_matrix_left1[idx] >= 4)[0][0]
            end = np.where(aln_matrix_left1[idx] >= 4)[0][-1]
            ax2.add_patch(mplpatches.Rectangle((start, idx - 0.5),end-start,1,hatch='\\',fill=False,snap=False, linewidth=1, edgecolor='black'))
        except IndexError:
            pass
    for idx in aux_dict_left1['low_mapq_idx']:
        try:
            start = np.where(aln_matrix_left1[idx] >= 0)[0][0]
            end = np.where(aln_matrix_left1[idx] >= 0)[0][-1]
            ax2.add_patch(mplpatches.Rectangle((start-0.5, idx - 0.5),end-start+1,1,hatch='\\',fill=True,snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
    ax2.set_xticks([0, window/2, window, window*1.5, 2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window*1.5)), str(int(window))])

    ### Right Breakpoint 1
    im = ax3.imshow(aln_matrix_right1, cmap=ListedColormap(colors), vmin=-1, vmax=5)
    ax3.axvline(x=window, color='black', linewidth=1, linestyle='--')
    for idx in aux_dict_right1['split_idx']:
        try:
            start = np.where(aln_matrix_right1[idx] >= 4)[0][0]
            end = np.where(aln_matrix_right1[idx] >= 4)[0][-1]
            ax3.add_patch(mplpatches.Rectangle((start, idx - 0.5),end-start,1,hatch='\\',fill=False,snap=False, linewidth=1, edgecolor='black'))
        except IndexError:
            pass
    for idx in aux_dict_right1['low_mapq_idx']:
        try:
            start = np.where(aln_matrix_right1[idx] >= 0)[0][0]
            end = np.where(aln_matrix_right1[idx] >= 0)[0][-1]
            ax3.add_patch(mplpatches.Rectangle((start-0.5, idx - 0.5),end-start+1,1,hatch='\\',fill=True,snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
    ax3.set_xticks([0, window/2, window, window*1.5, 2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window*1.5)), str(int(window))])


    ### Left Breakpoint 2
    ax5.imshow(aln_matrix_left2, cmap=ListedColormap(colors), vmin=-1, vmax=5)
    ax5.axvline(x=window, color='black', linewidth=1, linestyle='--')
    for idx in aux_dict_left2['split_idx']:
        try:
            start = np.where(aln_matrix_left2[idx] >= 4)[0][0]
            end = np.where(aln_matrix_left2[idx] >= 4)[0][-1]
            ax5.add_patch(mplpatches.Rectangle((start, idx - 0.5),end-start,1,hatch='\\',fill=False,snap=False, linewidth=1, edgecolor='black'))
        except IndexError:
            pass
    for idx in aux_dict_left2['low_mapq_idx']:
        try:
            start = np.where(aln_matrix_left2[idx] >= 0)[0][0]
            end = np.where(aln_matrix_left2[idx] >= 0)[0][-1]
            ax5.add_patch(mplpatches.Rectangle((start-0.5, idx - 0.5),end-start+1,1,hatch='\\',fill=True,snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
    ax5.set_xticks([0, window/2, window, window*1.5, 2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window*1.5)), str(int(window))])


    ### Right Breakpoint 2
    im = ax6.imshow(aln_matrix_right2, cmap=ListedColormap(colors), vmin=-1, vmax=5)
    ax6.axvline(x=window, color='black', linewidth=1, linestyle='--')
    for idx in aux_dict_right2['split_idx']:
        try:
            start = np.where(aln_matrix_right2[idx] >= 4)[0][0]
            end = np.where(aln_matrix_right2[idx] >= 4)[0][-1]
            ax6.add_patch(mplpatches.Rectangle((start, idx - 0.5),end-start,1,hatch='\\',fill=False,snap=False, linewidth=1, edgecolor='black'))
        except IndexError:
            pass
    for idx in aux_dict_right2['low_mapq_idx']:
        try:
            start = np.where(aln_matrix_right2[idx] >= 0)[0][0]
            end = np.where(aln_matrix_right2[idx] >= 0)[0][-1]
            ax6.add_patch(mplpatches.Rectangle((start-0.5, idx - 0.5),end-start+1,1,hatch='\\',fill=True,snap=False, linewidth=1, edgecolor='none', facecolor='grey', alpha=0.5))
        except IndexError:
            pass
    ax6.set_xticks([0, window/2, window, window*1.5, 2*window], labels=[str(-int(window)), str(-int(window/2)), '0', str(int(window*1.5)), str(int(window))])

    
    cbar = fig.colorbar(im, cmap=ListedColormap(colors[1:]), ax=[ax1, ax2, ax3, ax4, ax5, ax6], shrink=0.5, ticks=[0.3,1.1,2,2.8,3.7,4.5])
    labels = ['M', 'I', 'D', 'N', 'S', 'H']
    cbar.ax.set_yticklabels(labels)
    if title != None:
        ax1.set_title(title)
    if outfile != None:
        plt.savefig(outfile)
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
