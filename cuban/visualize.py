import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import numpy as np
import pysam
from scipy.signal import savgol_filter

from .utils import compute_aln_matrix, pad_alignment_matrices, compute_rep_df, compute_isize_orientation_dict, add_comma_to_pos, reorder_by_hp
from . import coverage as coverage_backend

rc_params = {'agg.path.chunksize': 1000000, 'hatch.linewidth': 0.5, 'font.weight': 'bold', 'axes.labelweight': 'bold'}


rep_y_pos_map = {'LINE' : (-6, '#6ac0b7'),
                 'SINE' : (-14, '#b7954b'),
                 'LTR' : (-22, '#f0b6a0'),
                 'DNA' : (-30, '#5066a2'),
                 'Simple_repeat' : (-38, '#504669'),
                 'Satellite' : (-46, '#df624c'),
                 'Low_complexity' : (-54, '#61856b'),
                 'Retroposon' : (-62, '#2f7155')}


# Strand barcode strip drawn inside the spacer between the two breakpoint matrices:
# blue cell per forward-strand row, red cell per reverse-strand row.
STRAND_STRIP_W = 3
_STRAND_COLORS = ['#8eb0d8', '#d8a098']  # 0 → forward (pastel blue), 1 → reverse (pastel red)


def _build_strand_strips(shape, aux_left, aux_right, window):
    """Build a masked (n_rows, n_cols) array that only paints the two strand strips
    inside the 50-col spacer; everywhere else is masked / transparent."""
    n_rows, n_cols = shape
    W = 2 * window
    strip = np.full((n_rows, n_cols), -1, dtype=int)

    def fill_side(aux, col_start):
        rev_set = set(aux['reverse_idx'])
        for i, name in enumerate(aux['name']):
            if isinstance(name, str) and name.startswith('__pad'):
                continue
            strip[i, col_start:col_start + STRAND_STRIP_W] = 1 if i in rev_set else 0

    fill_side(aux_left,  W + 1)
    fill_side(aux_right, W + 50 - STRAND_STRIP_W - 1)
    return np.ma.masked_equal(strip, -1)


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
            indices = np.where((aln_matrix[idx] == 4) | (aln_matrix[idx] == 5))[0]
            intervals = []
            if len(indices) > 0:
                start = indices[0]
                for i in range(1, len(indices)):
                    if indices[i] != indices[i-1] + 1:
                        intervals.append((start, indices[i-1]))
                        start = indices[i]
                intervals.append((start, indices[-1]))
                
                for interval in intervals:
                    start = interval[0]
                    end = interval[1]
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
        
        
def _robust_savgol_window(n: int) -> int:
    """ Computes an odd Savitzky-Golay window size from the coverage frame length `n`:
    ~5% of n (minimum 5), forced odd, capped at n-1 (kept odd), minimum 3. Robust to the
    short frames produced by binned (large-SV) coverage, unlike a window derived from raw
    SV size in bp. """
    raw_w = max(5, int(n * 0.05))
    w = raw_w if raw_w % 2 == 1 else raw_w + 1
    w = min(w, n - 1)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def gather_data(sv_type: str, bam_name: str, chrom: str, start: int, end: int, window: int, padding: int, collapse_ins: bool, rep_df: pd.DataFrame, coverage_dir: str=None, cache_dir: str=None, max_reads: int=5000, downsample: str='early_stop', bin_size: int=1) -> dict:
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
        coverage_dir (str, optional): Directory with precomputed mosdepth output for this sample. Defaults to None.
        cache_dir (str, optional): Directory to cache mosdepth output in when it has to be computed. Defaults to None.
        max_reads (int, optional): Maximum number of reads to include per alignment matrix. Defaults to 5000.
        downsample (str, optional): 'early_stop' or 'random' downsampling strategy. Defaults to 'early_stop'.
        bin_size (int, optional): Coverage bin size in bp; > 1 averages coverage into fixed-width
            windows (for large SVs) and skips insert-size/orientation computation, which is too
            costly and uninformative at that scale. Defaults to 1 (per-base).

    Returns:
        dict: Dictionary containing the alignment matrix, coverage, repeat overlap, insert size outliers and discordant orientation.
    """

    ### Initialize result dictionary
    result = dict()

    ### Load BAM file
    bam = pysam.AlignmentFile(bam_name, 'rb')

    ### Compute alignment matrix
    if sv_type in ['DEL', 'DUP', 'INV']:

        aln_matrix_left, aux_dict_left = compute_aln_matrix(bam, chrom, start - window, start + window, collapse_ins=collapse_ins, max_reads=max_reads, downsample=downsample)
        aln_matrix_right, aux_dict_right = compute_aln_matrix(bam, chrom, end - window, end + window, collapse_ins=collapse_ins, max_reads=max_reads, downsample=downsample)
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
        if end - start < 200000000:
            compute_cov = True
        else:
            compute_cov = False
        result['compute_cov'] = compute_cov
        
        ### Compute coverage
        if compute_cov:
            cov_padding = max(padding, int((end - start) * 0.2))
            cov_start = max(1, start - cov_padding)
            cov, cov_minq = coverage_backend.get_coverage(bam_name, chrom, cov_start, end + cov_padding, coverage_dir=coverage_dir, cache_dir=cache_dir, bin_size=bin_size)

            result['cov_padding'] = cov_padding
            result['left_padding'] = start - cov_start
            result['cov'] = cov
            result['cov_minq'] = cov_minq
            if bin_size > 1 or 1000 < end - start < 100000:
                w = _robust_savgol_window(len(cov_minq))
                if len(cov_minq) > w:
                    result['cov_minq_smoothed'] = savgol_filter(cov_minq[2], w, 3)

            ### Compute insert size outliers and discordant orientation (skipped when binned:
            ### too costly and uninformative at that scale)
            if bin_size <= 1:
                isize_orient_dict = compute_isize_orientation_dict(bam_name, chrom, cov_start, end + cov_padding)
                result['isize_orient_dict'] = isize_orient_dict

        ### Compute repeat overlap
        rep_df_sample = compute_rep_df(rep_df, chrom, start, end, padding=padding)
        result['rep_df'] = rep_df_sample
        
    elif sv_type in ['INS', 'BND']:
        
        ### Compute alignment matrix
        aln_matrix, aux_dict = compute_aln_matrix(bam, chrom, start - window, end + window, collapse_ins=collapse_ins, max_reads=max_reads, downsample=downsample)
        result['aln_matrix'] = aln_matrix
        result['aux_dict'] = aux_dict

        ### Compute coverage
        result['compute_cov'] = True
        cov_padding = padding
        cov_start = max(1, start - cov_padding)
        cov, cov_minq = coverage_backend.get_coverage(bam_name, chrom, cov_start, end + cov_padding, coverage_dir=coverage_dir, cache_dir=cache_dir, bin_size=bin_size)

        result['cov_padding'] = cov_padding
        result['left_padding'] = start - cov_start
        result['cov'] = cov
        result['cov_minq'] = cov_minq
        if bin_size > 1 or 1000 < end - start < 100000:
            w = _robust_savgol_window(len(cov_minq))
            if len(cov_minq) > w:
                result['cov_minq_smoothed'] = savgol_filter(cov_minq[2], w, 3)

        ### Compute insert size outliers and discordant orientation (skipped when binned:
        ### too costly and uninformative at that scale)
        if bin_size <= 1:
            isize_orient_dict = compute_isize_orientation_dict(bam_name, chrom, cov_start, end + cov_padding)
            result['isize_orient_dict'] = isize_orient_dict
        
        ### Compute repeat overlap
        rep_df_sample = compute_rep_df(rep_df, chrom, start, end, padding=padding)
        result['rep_df'] = rep_df_sample
        
    return result


def plot_cov(start: int, end: int, data: dict, ax_cov_ill: plt.Axes, padding: int, baseline_cov: float=None, plot_label: bool=True):
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
    if 'cov_minq_smoothed' in data:
        cov_minq_smoothed = data['cov_minq_smoothed']
        ax_cov_ill.plot(cov_minq[1], cov_minq_smoothed, color='black')
    if baseline_cov is not None:
        ax_cov_ill.axhline(y=baseline_cov, color='#df624c', linewidth=1, linestyle='--')
    ax_cov_ill.fill_between(cov[1], cov[2], color="grey", alpha=0.2)
    ax_cov_ill.axvline(x=data['left_padding'], color='black', linewidth=1, linestyle='--')
    ax_cov_ill.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    ax_cov_ill.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax_cov_ill.set_xticks([])
    yticks = ax_cov_ill.get_yticks()
    yticks_filtered = yticks[yticks >= 0]
    ax_cov_ill.set_yticks(yticks_filtered)
    if baseline_cov is not None and baseline_cov > 0:
        top = min(yticks[-1], 4 * baseline_cov)
    else:
        top = yticks[-1]
    top = max(top, 1)
    # The repeat track lives below y=0 (rep_y_pos_map spans 0..-70 in the
    # historical ~90x-top layout); scale that band with the actual coverage
    # range so shallow BAMs don't squash the coverage line into a sliver.
    ax_cov_ill.set_ylim(bottom=-70 * (top / 90), top=top)
    if plot_label:
        ax_cov_ill.text(0, 1.1, 'Coverage', transform=ax_cov_ill.transAxes, ha='left',
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
    
    ### Repeat rows scale with the coverage axis (see plot_cov's ylim comment)
    bottom = ax_cov_ill.get_ylim()[0]
    scale = -bottom / 70 if bottom < 0 else 1.0

    ### Plot
    for i in range(len(rep_df)):
        try:
            ax_cov_ill.hlines(y=rep_y_pos_map[rep_df.loc[i, 'repClass']][0] * scale, xmin=rep_df.loc[i, 'genoStart'], xmax=rep_df.loc[i, 'genoEnd'], linewidth=4, color=rep_y_pos_map[rep_df.loc[i, 'repClass']][1])
        except KeyError: # repeat type not in rep_y_pos_map
            continue

    ### Add repeat labels
    for key in rep_y_pos_map.keys():
        ax_cov_ill.text(10, (rep_y_pos_map[key][0]-2) * scale, key.replace('_', ' '), fontsize=7, horizontalalignment='left', verticalalignment='center', color='black', weight='bold')
        

def _plot_not_computed_panel(ax: plt.Axes, data: dict, padding: int, plot_label: bool, label: str):
    """ Renders an empty panel with a small centered note, used in place of the insert-size /
    orientation histograms when 'isize_orient_dict' was not computed (binned/large-SV mode). """
    cov = data['cov']
    ax.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axvline(x=data['left_padding'], color='black', linewidth=1, linestyle='--')
    ax.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    ax.text(0.5, 0.5, 'not computed at this scale', transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='grey', style='italic')
    if plot_label:
        ax.text(0, 1.05, label, transform=ax.transAxes, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))


def plot_isize(data: dict, ax_isize_ill: plt.Axes, padding: int, plot_label: bool=True):
    """ Plots insert size outliers.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_isize_ill (plt.Axes): Axes to plot on
        padding (int): Padding around the SV to compute coverage.
    """    
    
    ### Extract data
    isize_orient_dict = data.get('isize_orient_dict')
    if isize_orient_dict is None:
        _plot_not_computed_panel(ax_isize_ill, data, padding, plot_label, 'Insert Size Outliers')
        return
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
    ax_isize_ill.axvline(x=data['left_padding'], color='black', linewidth=1, linestyle='--')
    ax_isize_ill.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    
    if plot_label:
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


def plot_orient(data: dict, ax_orient_ill: plt.Axes, padding: int, plot_label: bool=True):
    """ Plots discordant orientation.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_orient_ill (plt.Axes): Axes to plot on
        padding (int): Padding around the SV to compute coverage.
    """    
    
    ### Extract data
    isize_orient_dict = data.get('isize_orient_dict')
    if isize_orient_dict is None:
        _plot_not_computed_panel(ax_orient_ill, data, padding, plot_label, 'Discordant Read Pairs')
        return
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
    num_tx = plot_hist(isize_orient_dict.get('tx', []), color='#df624c', linestyle='--')
    
    ### Set plot limits and labels
    ax_orient_ill.set_xlim(left=0, right=cov.iloc[-1, 1])
    ax_orient_ill.axvline(x=data['left_padding'], color='black', linewidth=1, linestyle='--')
    ax_orient_ill.axvline(x=cov.iloc[-1, 1] - padding, color='black', linewidth=1, linestyle='--')
    
    if plot_label:
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
    
    
def _hp_label(hp):
    return 'Unassigned' if hp == -1 else f'HP:{hp}'


def _has_hp_info(aux_dict):
    return any(v != -1 for v in aux_dict.get('haplotag_idx', []))


def _render_dual_panel(ax, aln_left, aux_left, aln_right, aux_right, colors, window, tech,
                       draw_header, header_y, draw_xticks, ylabel=None):
    """Renders one dual-breakpoint CIGAR strip on `ax`: pad+concat the two matrices,
    imshow, overlays, breakpoint axvlines, intra-axis read connections, header text."""
    aln_left_p, aln_right_p = pad_alignment_matrices(aln_left, aln_right)
    spacer = np.ones((aln_left_p.shape[0], 50)) * -1
    aln_concat = np.concatenate((aln_left_p, spacer, aln_right_p), axis=1)
    ax.imshow(aln_concat, cmap=ListedColormap(colors), vmin=-1, vmax=8)
    ax.imshow(_build_strand_strips(aln_concat.shape, aux_left, aux_right, window),
              cmap=ListedColormap(_STRAND_COLORS), vmin=0, vmax=1)

    # Left breakpoint
    ax.axvline(x=window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_left, aln_left_p, ax)
    if tech == 'ill':
        add_disco_overlay(aux_left, aln_left_p, ax, 'rr')
        add_disco_overlay(aux_left, aln_left_p, ax, 'ff')
        add_disco_overlay(aux_left, aln_left_p, ax, 'rf')
        add_disco_overlay(aux_left, aln_left_p, ax, 'tx')
    add_splitread_overlay(aux_left, aln_left_p, ax)

    # Right breakpoint
    ax.axvline(x=(window + 50) + 2 * window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux_right, aln_right_p, ax, offset=2 * window + 50)
    if tech == 'ill':
        add_disco_overlay(aux_right, aln_right_p, ax, 'rr', offset=2 * window + 50)
        add_disco_overlay(aux_right, aln_right_p, ax, 'ff', offset=2 * window + 50)
        add_disco_overlay(aux_right, aln_right_p, ax, 'rf', offset=2 * window + 50)
        add_disco_overlay(aux_right, aln_right_p, ax, 'tx', offset=2 * window + 50)
    add_splitread_overlay(aux_right, aln_right_p, ax, offset=2 * window + 50)

    # Read connections (merge on read name within this axis only)
    df_l = pd.DataFrame(aux_left['name'], columns=['name']).reset_index()
    df_r = pd.DataFrame(aux_right['name'], columns=['name']).reset_index()
    df_m = df_l.merge(df_r, on='name', how='left', suffixes=('_left', '_right'))
    df_m = df_m[df_m['index_right'].notna()].reset_index(drop=True)
    if len(df_m) > 0:
        df_m['index_right'] = df_m['index_right'].astype(int)
        df_m['split_left'] = 0
        df_m.loc[df_m['index_left'].isin(aux_left['split_idx']), 'split_left'] = 1
        df_m['split_right'] = 0
        df_m.loc[df_m['index_right'].isin(aux_right['split_idx']), 'split_right'] = 1
        left_strip_inner = 2 * window + STRAND_STRIP_W + 1 - 0.5
        right_strip_inner = 2 * window + 50 - STRAND_STRIP_W - 1 - 0.5
        for i in range(len(df_m)):
            both_split = df_m.loc[i, 'split_left'] == 1 and df_m.loc[i, 'split_right'] == 1
            color = 'black' if (tech == 'pb' or both_split) else '#df624c'
            ax.plot([left_strip_inner, right_strip_inner],
                    [df_m.loc[i, 'index_left'], df_m.loc[i, 'index_right']],
                    color=color, linewidth=0.5, linestyle='dotted')

    # Spacer guides
    ax.axvline(x=(window * 2) - 0.5, color='lightgrey', linewidth=1, linestyle='--')
    ax.axvline(x=(window * 2) + 49.5, color='lightgrey', linewidth=1, linestyle='--')

    if draw_xticks:
        ax.set_xticks([0, window / 2, window, window * 1.5, 2 * window,
                       2 * window + 50, 2 * window + 50 + window / 2, 2 * window + 50 + window,
                       2 * window + 50 + window * 1.5, 2 * window + 50 + 2 * window],
                      labels=[str(-int(window)), str(-int(window / 2)), '0', str(int(window / 2)), str(int(window)),
                              str(-int(window)), str(-int(window / 2)), '0', str(int(window / 2)), str(int(window))])
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=0, ha='right', va='center', labelpad=20, fontsize=8)

    if draw_header:
        bbox = dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3)
        ax.text(0.185, header_y, 'Reads Left Breakpoint', transform=ax.transAxes, ha='left', bbox=bbox)
        ax.text(0.815, header_y, 'Reads Right Breakpoint', transform=ax.transAxes, ha='right', bbox=bbox)
        ax.text(0.535, header_y, 'Read Connections', transform=ax.transAxes, ha='right', bbox=bbox)


def _render_single_panel(ax, aln, aux, colors, window, tech,
                         draw_header, header_y, draw_xticks, ylabel=None):
    """Renders one single-breakpoint CIGAR strip on `ax` (INS case)."""
    ax.imshow(aln, cmap=ListedColormap(colors), vmin=-1, vmax=8)
    ax.axvline(x=window, color='black', linewidth=1, linestyle='--')
    add_mapq_overlay(aux, aln, ax)
    if tech == 'ill':
        add_disco_overlay(aux, aln, ax, 'rr')
        add_disco_overlay(aux, aln, ax, 'ff')
        add_disco_overlay(aux, aln, ax, 'rf')
        add_disco_overlay(aux, aln, ax, 'tx')
    add_splitread_overlay(aux, aln, ax)

    if draw_xticks:
        ax.set_xticks([0, window / 2, window, window * 1.5, 2 * window],
                      labels=[str(-int(window)), str(-int(window / 2)), '0', str(int(window / 2)), str(int(window))])
    else:
        ax.set_xticks([])
    ax.set_yticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=0, ha='right', va='center', labelpad=20, fontsize=8)

    if draw_header and tech == 'pb':
        ax.text(0.0, header_y, 'Reads Around Breakpoint', transform=ax.transAxes, ha='left',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', linewidth=1.3))


def _annotate_hp_bands(ax, bands):
    """Place left-side text labels at the vertical center of each HP band."""
    for hp, y_center, _n_total in bands:
        ax.text(-0.005, y_center, _hp_label(hp),
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', rotation=90, fontsize=8)


def plot_cigar(sv_type: str, data: dict, ax_cig_ill: plt.Axes, colors: list, window: int, tech: str, dataB: dict=None):
    """ Plots CIGAR string information around breakpoints.

    Args:
        data (dict): Sequencing data, result of gather_data_ill
        ax_cig_ill (plt.Axes): Axes to plot on
        colors (list): List of colors for the alignment matrix
        window (int): Window around each breakpoint to collect CIGAR string information for.
    """
    header_y = 1.05 if tech == 'ill' else 1.2

    if sv_type in ['DEL', 'DUP', 'INV', 'BND']:

        ### Extract data
        if sv_type != 'BND':
            aux_dict_left = data['aux_dict_left']
            aln_matrix_left = data['aln_matrix_left']
            aux_dict_right = data['aux_dict_right']
            aln_matrix_right = data['aln_matrix_right']
        else:
            aux_dict_left = data['aux_dict']
            aln_matrix_left = data['aln_matrix']
            aux_dict_right = dataB['aux_dict']
            aln_matrix_right = dataB['aln_matrix']

        # For DEL/DUP/INV, gather_data pre-pads left/right to equal height while aux_dicts
        # stay at raw size; trim back to raw before any per-read row processing.
        aln_matrix_left = aln_matrix_left[:len(aux_dict_left['haplotag_idx'])]
        aln_matrix_right = aln_matrix_right[:len(aux_dict_right['haplotag_idx'])]

        bands = None
        if _has_hp_info(aux_dict_left) or _has_hp_info(aux_dict_right):
            aln_matrix_left, aux_dict_left, aln_matrix_right, aux_dict_right, bands = \
                reorder_by_hp(aln_matrix_left, aux_dict_left,
                              aln_matrix_right, aux_dict_right)

        _render_dual_panel(ax_cig_ill, aln_matrix_left, aux_dict_left,
                           aln_matrix_right, aux_dict_right,
                           colors, window, tech,
                           draw_header=True, header_y=header_y, draw_xticks=True)
        if bands is not None:
            _annotate_hp_bands(ax_cig_ill, bands)

    elif sv_type == 'INS':

        aln_matrix = data['aln_matrix']
        aux_dict = data['aux_dict']

        bands = None
        if _has_hp_info(aux_dict):
            aln_matrix, aux_dict, _r1, _r2, bands = reorder_by_hp(aln_matrix, aux_dict)

        _render_single_panel(ax_cig_ill, aln_matrix, aux_dict, colors, window, tech,
                             draw_header=True, header_y=header_y, draw_xticks=True)
        if bands is not None:
            _annotate_hp_bands(ax_cig_ill, bands)
        
        
def cuban(samples: dict, rep_df: pd.DataFrame, sv_type: str, chrom: str, start: int, end: int, padding: int=1500, window: int=100, collapse_ins: bool=True, outfile: str=None, sv_len: int=None, cache_dir: str=None, max_reads: int=5000, downsample: str='early_stop', bin_size: int=1):
    """ Visualizes alignment information around a structural variant for one or multiple samples.

    Args:
        samples (dict): Dictionary containing sample information. Keys are sample names and values are dictionaries containing family status, disease status, BAM filename and baseline coverage.
            'baseline_cov' may be a float or the string 'auto' (resolved per-chromosome at render time).
            'coverage_dir', if present, points at a precomputed mosdepth output directory for that sample.
        rep_df (pd.DataFrame): Repeat dataframe
        sv_type (str): Type of the structural variant
        chrom (str): Chromosome
        start (int): Start position
        end (int): End position
        padding (int, optional): Padding around the SV to compute coverage.. Defaults to 1500.
        window (int, optional): Window around each breakpoint to collect CIGAR string information for.. Defaults to 100.
        collapse_ins (bool, optional): Collapse insertions into one base.. Defaults to True.
        bin_size (int, optional): Coverage bin size in bp; > 1 averages coverage into fixed-width
            windows and skips insert-size/orientation panels (large-SV mode). Defaults to 1.
        outfile (str, optional): Output filename. Defaults to None.
        cache_dir (str, optional): Directory to cache mosdepth output in when it has to be computed. Defaults to None.
        max_reads (int, optional): Maximum number of reads to include per alignment matrix. Defaults to 5000.
        downsample (str, optional): 'early_stop' or 'random' downsampling strategy. Defaults to 'early_stop'.
    """
    
    ### Set up figure
    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c', 'lightgrey', 'lightgrey']
    num_samples = len(samples)
    num_rows = 0
    height_ratios = []
    if not samples:
        raise ValueError('No samples provided.')
    for sample in samples:
        technology = samples[sample]['technology']
        if technology == 'ill':
            num_rows += 5
            height_ratios.extend([1,3,0.5,0.5,7])
        elif technology == 'pb':
            num_rows += 3
            height_ratios.extend([1,3,7])
        else:
            raise ValueError(f"Sample '{sample}' has invalid technology '{technology}': technology must be 'ill' or 'pb'")
        
    with plt.style.context('ggplot'), mpl.rc_context(rc_params):
        fig = plt.figure(figsize=(25, 11 * num_samples))
        gs = gridspec.GridSpec(num_rows, 4, height_ratios=height_ratios, hspace=0.2, wspace=0.5, figure=fig)
    
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
            coverage_dir = samples[sample].get('coverage_dir')

            if baseline_cov == 'auto':
                baseline_cov = coverage_backend.get_baseline(bam_name, chrom, coverage_dir=coverage_dir, cache_dir=cache_dir)

            ### Extract Sequencing Data
            data = gather_data(sv_type, bam_name, chrom, start, end, window, padding, collapse_ins, rep_df, coverage_dir=coverage_dir, cache_dir=cache_dir, max_reads=max_reads, downsample=downsample, bin_size=bin_size)

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
                plot_cigar(sv_type, data, ax_cig_ill, colors, window, 'ill')
            
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
                plot_cigar(sv_type, data, ax_cig_pb, colors, window, 'pb')
            
                ### Update index
                i += 3
        
            ### Set title
            if not plotted_main_title:
                if sv_len is None:
                    sv_len = end - start
                try:
                    ax_title.text(0.5, 0.5, sv_type + ' ' + chrom + ':' + add_comma_to_pos(start) + '-' + add_comma_to_pos(end) + ' (' + add_comma_to_pos(sv_len) + ' bp)', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
                except ValueError:
                    ax_title.text(0.5, 0.5, sv_type + ' ' + chrom + ':' + add_comma_to_pos(start) + '-' + add_comma_to_pos(end) + ' (Unknown Length)', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
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
                plt.close()
        else:
            plt.show()
        
        
def cuban_bnd(samples: dict, rep_df: pd.DataFrame, chromA: str, startA: int, endA: int, chromB: str, startB: int, endB: int, padding: int=1500, window: int=100, collapse_ins: bool=True, outfile: str=None, cache_dir: str=None, max_reads: int=5000, downsample: str='early_stop', bin_size: int=1):
    """ Visualizes alignment information around an SV for one or multiple samples for two loci independently.

    Args:
        samples (dict): Dictionary containing sample information. Keys are sample names and values are dictionaries containing family status, disease status, BAM filename and baseline coverage.
            'baseline_cov' may be a float or the string 'auto' (resolved separately for chromA and chromB at render time).
            'coverage_dir', if present, points at a precomputed mosdepth output directory for that sample.
        rep_df (pd.DataFrame): Repeat dataframe
        chromA (str): Chromosome of first locus
        startA (int): Start position of first locus
        endA (int): End position of first locus
        chromB (str): Chromosome of second locus
        startB (int): Start position of second locus
        endB (int): End position of second locus
        padding (int, optional): Padding around the SV to compute coverage.. Defaults to 1500.
        window (int, optional): Window around each breakpoint to collect CIGAR string information for.. Defaults to 100.
        collapse_ins (bool, optional): Collapse insertions into one base.. Defaults to True.
        bin_size (int, optional): Coverage bin size in bp; > 1 averages coverage into fixed-width
            windows and skips insert-size/orientation panels (large-SV mode). Defaults to 1.
        outfile (str, optional): Output filename. Defaults to None.
        cache_dir (str, optional): Directory to cache mosdepth output in when it has to be computed. Defaults to None.
        max_reads (int, optional): Maximum number of reads to include per alignment matrix. Defaults to 5000.
        downsample (str, optional): 'early_stop' or 'random' downsampling strategy. Defaults to 'early_stop'.
    """
    # Set SV type
    sv_type = 'BND' 
    
    ### Set up figure
    colors = ['white', 'lightgrey', '#b7954b', '#5066a2', '#f0b6a0', '#6ac0b7', '#df624c', 'lightgrey', 'lightgrey']
    num_samples = len(samples)
    num_rows = 0
    height_ratios = []
    if not samples:
        raise ValueError('No samples provided.')
    for sample in samples:
        technology = samples[sample]['technology']
        if technology == 'ill':
            num_rows += 5
            height_ratios.extend([1,3,0.5,0.5,7])
        elif technology == 'pb':
            num_rows += 3
            height_ratios.extend([1,3,7])
        else:
            raise ValueError(f"Sample '{sample}' has invalid technology '{technology}': technology must be 'ill' or 'pb'")
        
    with plt.style.context('ggplot'), mpl.rc_context(rc_params):
        fig = plt.figure(figsize=(25, 11 * num_samples))
        gs = gridspec.GridSpec(num_rows, 4, height_ratios=height_ratios, hspace=0.5, wspace=0.35, figure=fig)
    
        ### Set up parameters
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
            coverage_dir = samples[sample].get('coverage_dir')

            if baseline_cov == 'auto':
                baseline_cov_a = coverage_backend.get_baseline(bam_name, chromA, coverage_dir=coverage_dir, cache_dir=cache_dir)
                baseline_cov_b = coverage_backend.get_baseline(bam_name, chromB, coverage_dir=coverage_dir, cache_dir=cache_dir)
            else:
                baseline_cov_a = baseline_cov_b = baseline_cov

            ### Extract Sequencing Data
            dataA = gather_data(sv_type, bam_name, chromA, startA, endA, window, padding, collapse_ins, rep_df, coverage_dir=coverage_dir, cache_dir=cache_dir, max_reads=max_reads, downsample=downsample, bin_size=bin_size)
            dataB = gather_data(sv_type, bam_name, chromB, startB, endB, window, padding, collapse_ins, rep_df, coverage_dir=coverage_dir, cache_dir=cache_dir, max_reads=max_reads, downsample=downsample, bin_size=bin_size)

            ### Define sample axes
            if technology == 'ill':
                axes_title = plt.subplot(gs[i, 0:4])
                axes_cov_ill = [plt.subplot(gs[i + 1, 0:2]), plt.subplot(gs[i + 1, 2:4])]
                axes_isize_ill = [plt.subplot(gs[i + 2, 0:2]), plt.subplot(gs[i + 2, 2:4])]
                axes_orient_ill = [plt.subplot(gs[i + 3, 0:2]), plt.subplot(gs[i + 3, 2:4])]
                axes_cig_ill = plt.subplot(gs[i + 4, 0:4])
                axes_split = [axes_cov_ill, axes_isize_ill, axes_orient_ill]
                axes = [axes_title, axes_cig_ill]
                for ax in axes_split:
                    for subax in ax:
                        subax.grid(False)
                        subax.set_facecolor('white')
                for ax in axes:
                    ax.grid(False)
                    ax.set_facecolor('white')
            
                ### Create plots for current sample, first locus
                if dataA['compute_cov']:
                    plot_cov(startA, endA, dataA, axes_cov_ill[0], padding, baseline_cov_a)
                    plot_rep(dataA, axes_cov_ill[0], rep_y_pos_map)
                    plot_isize(dataA, axes_isize_ill[0], padding)
                    plot_orient(dataA, axes_orient_ill[0], padding)
            
                ### Create plots for current sample, second locus
                if dataB['compute_cov']:
                    plot_cov(startB, endB, dataB, axes_cov_ill[1], padding, baseline_cov_b, plot_label=False)
                    plot_rep(dataB, axes_cov_ill[1], rep_y_pos_map)
                    plot_isize(dataB, axes_isize_ill[1], padding, plot_label=False)
                    plot_orient(dataB, axes_orient_ill[1], padding, plot_label=False)
                
                ### Plot joint alignment matrix
                plot_cigar(sv_type, dataA, axes_cig_ill, colors, window, 'ill', dataB)
            
                ### Update index
                i += 5
            
            elif technology == 'pb':
                axes_title = plt.subplot(gs[i, 0:4])
                axes_cov_pb = [plt.subplot(gs[i + 1, 0:2]), plt.subplot(gs[i + 1, 2:4])]
                axes_cig_pb = plt.subplot(gs[i + 2, 0:4])
                axes_split = [axes_cov_pb]
                axes = [axes_title, axes_cig_pb]
                for ax in axes_split:
                    for subax in ax:
                        subax.grid(False)
                        subax.set_facecolor('white')
                for ax in axes:
                    ax.grid(False)
                    ax.set_facecolor('white')
            
                ### Create plots for current sample, first locus
                if dataA['compute_cov']:
                    plot_cov(startA, endA, dataA, axes_cov_pb[0], padding, baseline_cov_a)
                    plot_rep(dataA, axes_cov_pb[0], rep_y_pos_map)
            
                ### Create plots for current sample, second locus
                if dataB['compute_cov']:
                    plot_cov(startB, endB, dataB, axes_cov_pb[1], padding, baseline_cov_b, plot_label=False)
                    plot_rep(dataB, axes_cov_pb[1], rep_y_pos_map)
                
                ### Plot joint alignment matrix
                plot_cigar(sv_type, dataA, axes_cig_pb, colors, window, 'pb', dataB)
            
                ### Update index
                i += 3
        
            ### Set title
            if not plotted_main_title:
                axes_title.text(0.5, 0.5, sv_type + ' ' + chromA + ':' + add_comma_to_pos(startA) + ' <> ' + chromB + ':' + add_comma_to_pos(startB), horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
                plotted_main_title = True
            axes_title.text(0.5, 0, name + ' (' + family_status.capitalize() + ', ' + disease_status.capitalize() + ')', horizontalalignment='center', verticalalignment='center', fontsize=12, weight='bold')
            axes_title.axis('off')
    
        ### Save figure
        if outfile != None:
            try:
                plt.savefig(outfile, dpi=96, bbox_inches='tight')
                plt.close()
            except OverflowError:
                plt.clf()
                plt.text(0.5, 0.5, 'Plotting not possible', ha='center', va='center')
                plt.savefig(outfile, dpi=96, bbox_inches='tight')
                plt.close()
        else:
            plt.show()

