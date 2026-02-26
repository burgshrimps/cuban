import io
import numpy as np
import pandas as pd
import pysam


def compute_overlap(s1, s2, e1, e2):
    """ Computes overlap between two segments. """
    return max(0, min(e1, e2) - max(s1, s2))


def cigartuples_to_array(cigartuples):
    """ Converts list of cigar tuples to numpy array. """
    cigar_array = []
    for cigartuple in cigartuples:
        cigar_array += [cigartuple[0]] * cigartuple[1]
    return np.array(cigar_array)


def compute_aln_matrix(bam, chrom, start, stop, collapse_ins=True):
    """ Computes alignment matrix consisting of CIGAR integers for a given region. """
    size = stop - start
    reads = []
    for read in bam.fetch(chrom, max(1, start), stop):
        if not read.is_unmapped:
            reads.append(read)

    aln_matrix = -1 * np.ones((len(reads), size))
    aux_dict = {'split_idx' : [],
                'low_mapq_idx' : [],
                'haplotag_idx': [],
                'name' : [],
                'discordant_idx_ff': [], 
                'discordant_idx_rr': [],
                'discordant_idx_rf': [],
                'discordant_idx_tx': []}

    for idx, read in enumerate(reads):
        
        if not read.is_unmapped and not read.is_duplicate and not read.is_qcfail: 
        
            aux_dict['name'].append(read.query_name)
            if read.has_tag('SA'):
                aux_dict['split_idx'].append(idx)
            if read.mapping_quality < 30:
                aux_dict['low_mapq_idx'].append(idx)
            if read.has_tag('HP'):
                aux_dict['haplotag_idx'].append(read.get_tag('HP'))
            else:
                aux_dict['haplotag_idx'].append(-1)
                
            if not read.mate_is_unmapped:
                if read.reference_id != read.next_reference_id:
                    aux_dict['discordant_idx_tx'].append(idx)
                
                elif read.is_reverse and read.mate_is_reverse:
                    aux_dict['discordant_idx_rr'].append(idx)
                    
                elif not read.is_reverse and not read.mate_is_reverse:
                    aux_dict['discordant_idx_ff'].append(idx)
                    
                elif not read.is_reverse and read.mate_is_reverse and read.template_length < 0:
                    aux_dict['discordant_idx_rf'].append(idx)
                            
                elif read.is_reverse and not read.mate_is_reverse and read.template_length > 0:
                    aux_dict['discordant_idx_rf'].append(idx)
                
            cigararray = cigartuples_to_array(read.cigartuples)

            if read.cigartuples[0][0] == 4 or read.cigartuples[0][0] == 5:
                read_start = read.reference_start - read.cigartuples[0][1]
            else:
                read_start = read.reference_start
            if read.cigartuples[-1][0] == 4 or read.cigartuples[-1][0] == 5:
                read_end = read.reference_end + read.cigartuples[-1][1]
            else:
                read_end = read.reference_end

            # Identify INS in CIGARARRAY and remove them for alignment visualisation
            if collapse_ins:
                ins_idx = np.argwhere(cigararray == 1).flatten()  # Find positions of INS (1) in CIGARARRAY
                diff = np.diff(ins_idx)  # Compute difference between consecutive INS positions
                idx_runs = np.argwhere(diff == 1).flatten()  # Find positions of consecutive INS positions
                mask = np.ones(len(cigararray), bool)
                mask[ins_idx[idx_runs]] = 0  # Deselect consecutive INS positions exept the first one
                mask[ins_idx - 1] = 0  # Deselect the position before the first INS position to avoid shifting of cigararray
                cigararray = cigararray[mask]  # Remove consecutive INS positions from CIGARARRAY
            
            if read_start < start and read_end > stop:
                # Read completely covers region
                if collapse_ins:
                    start_idx_read = start - read_start
                else:
                    start_idx_read = np.argwhere(cigararray != 1).flatten()[start - read_start]  # shift start by number of INS
                start_idx_aln = 0
                end_idx_read = start_idx_read + size
                end_idx_aln = size
            
            elif read_start < start and read_end <= stop:
                # Read is partially contained in region
                if collapse_ins:
                    start_idx_read = start - read_start
                else:
                    start_idx_read = np.argwhere(cigararray != 1).flatten()[start - read_start]
                start_idx_aln = 0
                if len(cigararray[start_idx_read:]) < size:
                    end_idx_read = len(cigararray)
                    end_idx_aln = end_idx_read - start_idx_read
                else:
                    end_idx_read = start_idx_read + size
                    end_idx_aln = size

            elif read_start > start and read_end > stop:
                # Read is partially contained in region
                start_idx_read = 0
                start_idx_aln = read_start - start 
                end_idx_read = start_idx_read + size - (read_start - start) 
                end_idx_aln = size

            else:
                # Read is fully contained in region
                start_idx_read = 0
                start_idx_aln = read_start - start
                if len(cigararray) < size:
                    end_idx_read = len(cigararray)
                    end_idx_aln = start_idx_aln + len(cigararray)
                else:
                    end_idx_read = start_idx_read + size - (read_start - start) 
                    end_idx_aln = size
                # if start_idx_aln < 0:
                #     print(start_idx_aln)
                
            if end_idx_aln > 0 and end_idx_read > 0:
                try:
                    aln_matrix[idx, start_idx_aln:end_idx_aln] = cigararray[start_idx_read:end_idx_read]
                except: 
                    pass
                    #print(idx, read.query_name, start_idx_aln, end_idx_aln, start_idx_read, end_idx_read, read_start, read_end, start, stop, size, len(cigararray),  np.sum(cigararray == 1))
    
    return aln_matrix, aux_dict


def pad_alignment_matrices(aln_matrix_left, aln_matrix_right):
    """ Pads alignment matrices with zeros to make them the same height. """
    if len(aln_matrix_right) < len(aln_matrix_left):
        aln_matrix_right = np.pad(aln_matrix_right, ((0, len(aln_matrix_left) - len(aln_matrix_right)), (0, 0)), 'constant', constant_values=(-1, -1))
    elif len(aln_matrix_right) > len(aln_matrix_left):
        aln_matrix_left = np.pad(aln_matrix_left, ((0, len(aln_matrix_right) - len(aln_matrix_left)), (0, 0)), 'constant', constant_values=(-1, -1))
    return aln_matrix_left, aln_matrix_right


def _compute_cov_df_binned_fast(bam_filename: str, chrom: str, start: int, stop: int,
                                 minq: int, bin_size: int):
    """Single-pass coverage computation for large SVs via bam.fetch().

    Replaces two pysam.depth() subprocess calls (each generating ~56 MB of pipe output
    for a 2.8 Mb region) with one native BAM scan that simultaneously computes both
    all-reads and high-MAPQ coverage. Per-base depth is approximated by accumulating
    each read's reference-span overlap into bins — accurate to within CIGAR-level
    deletion noise, negligible at bin_size >= 100 bp resolution.

    Returns:
        (cov, cov_minq): DataFrames with columns [0=chrom, 1=relative_pos, 2=mean_depth]
                         — same schema as compute_cov_df().
    """
    region_len = stop - start
    n_bins = max(1, (region_len + bin_size - 1) // bin_size)
    cov_sum  = np.zeros(n_bins, dtype=np.int64)
    covq_sum = np.zeros(n_bins, dtype=np.int64)

    with pysam.AlignmentFile(bam_filename, 'rb') as bam:
        for read in bam.fetch(chrom, start, stop):
            if (read.is_unmapped or read.is_secondary
                    or read.is_supplementary or read.is_duplicate
                    or read.reference_end is None):
                continue
            # Relative coordinates, clipped to window
            rstart = max(read.reference_start, start) - start
            rend   = min(read.reference_end,   stop)  - start
            if rstart >= rend:
                continue

            high_q  = read.mapping_quality >= minq
            b_first = rstart // bin_size
            b_last  = min((rend - 1) // bin_size, n_bins - 1)

            for b in range(b_first, b_last + 1):
                bpos_s  = b * bin_size
                bpos_e  = min(bpos_s + bin_size, region_len)
                overlap = min(rend, bpos_e) - max(rstart, bpos_s)
                if overlap > 0:
                    cov_sum[b] += overlap
                    if high_q:
                        covq_sum[b] += overlap

    positions = np.arange(n_bins, dtype=np.int64) * bin_size
    cov_mean  = np.round(cov_sum  / bin_size).astype(np.int32)
    covq_mean = np.round(covq_sum / bin_size).astype(np.int32)

    cov      = pd.DataFrame({0: chrom, 1: positions, 2: cov_mean})
    cov_minq = pd.DataFrame({0: chrom, 1: positions, 2: covq_mean})
    return cov, cov_minq


def compute_cov_df(bam_filename, chrom, start, stop, minq=30, bin_size=1):
    """ Computes coverage for given region. Coverage is computed twice. Once for all reads and once for reads with
    mapping quality >= minq. If bin_size > 1, coverage is binned into fixed-width windows before returning. """
    
    if bin_size > 1:
        return _compute_cov_df_binned_fast(bam_filename, chrom, start, stop, minq, bin_size)

    # Small SV path: samtools depth subprocess (per-base, accurate)
    region = f'{chrom}:{start}-{stop}'
    cov_str      = pysam.depth(bam_filename, '-r', region, '-a')
    cov_minq_str = pysam.depth(bam_filename, '-r', region, '-a', '-Q', str(minq))

    cov      = pd.read_csv(io.StringIO(cov_str),      sep='\t', header=None, names=[0, 1, 2],
                           dtype={1: int, 2: int})
    cov_minq = pd.read_csv(io.StringIO(cov_minq_str), sep='\t', header=None, names=[0, 1, 2],
                           dtype={1: int, 2: int})

    cov[1]      = cov[1]      - cov.loc[0, 1]
    cov_minq[1] = cov_minq[1] - cov_minq.loc[0, 1]
    return cov, cov_minq


def bin_coverage(df: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """Bins a per-base coverage DataFrame into fixed-width windows.

    Groups rows by (position // bin_size) * bin_size, computes mean
    coverage per bin (rounded to int), and returns a DataFrame with the
    same three-column schema: col 0 = chrom, col 1 = bin start position,
    col 2 = mean coverage depth.

    Args:
        df (pd.DataFrame): Per-base coverage DataFrame as returned by
            compute_cov_df (columns 0, 1, 2).
        bin_size (int): Width of each bin in bases. A value of 1 returns
            df unchanged (no-op).

    Returns:
        pd.DataFrame: Binned coverage DataFrame, same column structure.
    """
    if bin_size <= 1:
        return df
    bin_col = (df[1] // bin_size) * bin_size
    binned = df.groupby(bin_col, sort=True).agg({0: 'first', 2: 'mean'}).reset_index()
    binned[2] = binned[2].round().astype(int)
    return binned[[0, 1, 2]].reset_index(drop=True)


def compute_rep_df(rep_df, chrom, start, stop, padding=500):
    """ Computes repeat overlap for a given region. """
    rep_df = rep_df[(rep_df['genoName'] == chrom) & (rep_df['genoStart'] >= start - padding) & (rep_df['genoEnd'] <= stop + padding)].copy()
    rep_df['genoStart'] = rep_df['genoStart'] - start + padding
    rep_df['genoEnd'] = rep_df['genoEnd'] - start + padding
    rep_df = rep_df[(rep_df['genoEnd'] > 0) & (rep_df['genoStart'] < stop - start + (2*padding))]
    rep_df.reset_index(drop=True, inplace=True)
    return rep_df

def get_variant_neighbourhood(df, chrom, start, stop, padding=500):
    """ Computes overlap with other SVs in the same region as the given SV. """
    
    df = df[(df['chrom'] == chrom) & (df['start'] >= start - padding) & (df['end'] <= stop + padding)].copy()
    df['start'] = df['start'] - start + padding
    df['end'] = df['end'] - start + padding
    df = df[(df['end'] > 0) & (df['start'] < stop - start + (2*padding))]
    df.reset_index(drop=True, inplace=True)
    return df

def get_read_by_name(bam, chrom, start, stop, read_name):
    """ Returns read with given name. """
    for read in bam.fetch(chrom, start-5, stop+5):
        if read.query_name == read_name:
            return read
    return None

def add_comma_to_pos(pos):
    """ Adds comma to position. """
    pos = str(int(pos))
    if len(pos) < 4:
        return pos
    newstr = ''
    for i in range(len(pos)):
        newstr = pos[::-1][i] + newstr
        if i % 3 == 2:
            newstr = ',' + newstr
    
    if newstr[0] == ',':
        newstr = newstr[1:]
    return newstr


def compute_baseline_cov(bam_filename: str, chrom: str, n: int=1000, s: int=1000, target_regions: pd.DataFrame=None) -> float:
    """ Computes baseline coverage for a given chromosome.

    Args:
        bam_filename (str): Filename of BAM file
        chrom (str): Chromosome
        n (int, optional): Number of regions to sample. Defaults to 1000.
        s (int, optional): Size of each sampled region. Defaults to 1000.

    Returns:
        float: Median coverage across sampled regions
    """    
    
    # Open BAM file
    bam = pysam.AlignmentFile(bam_filename, 'rb')
    
    # Get chromosome index to get chromosome length
    if chrom == 'chrX':
        chrom_idx = 22
    elif chrom == 'chrY':
        chrom_idx = 23
    elif chrom == 'chrM':
        chrom_idx = 24
    else:
        chrom_idx = int(chrom[3:]) - 1

    # for targeted sequence randomly select at most 100 regions from the current chromosome
    exome=False
    if target_regions is not None:
        target_regions_chrom = target_regions.loc[target_regions['chr'] == chrom]
        # default to standard coverage computation if there are no target regions in the current chromosome
        if not target_regions_chrom.empty:
            exome=True
            regions_df = target_regions_chrom.sample(min(n,target_regions_chrom.shape[0]), replace=False, axis=0, random_state=1)
            regions = regions_df.apply(lambda x: [x['chr'],x['start'],min(x['end'],x['start']+s)],axis=1)

    if not exome:
        regions = []
        for i in range(n):
            if len(bam.lengths)>1:
                start = np.random.randint(0, bam.lengths[chrom_idx])
            else:
                start = np.random.randint(0, bam.lengths[0])
            stop = start + s
            regions.append([chrom, start, stop])

    baseline_coverage_mean = []
    for region in regions:
        chrom, start, stop = region
        coverage = [0] * (stop - start + 1)
        for pileupcolumn in bam.pileup(chrom, start, stop, min_mapping_quality=20, flag_filter=1540, stepper='samtools', ignore_orphans=False, ignore_overlaps=False):
            if start <= pileupcolumn.pos <= stop:
                coverage[pileupcolumn.pos - start] = pileupcolumn.nsegments
        
        baseline_coverage_mean.append(np.mean(coverage))
        
    return np.median(baseline_coverage_mean)


def compute_isize_orientation_dict(bam_filename: str, chrom: str, start: int, end: int, thr_min: int=50, thr_max: int=1000) -> dict:
    """ Computes insert size distribution for a given region.

    Args:
        bam_filename (str): Filename of BAM file
        chrom (str): Chromosome
        start (int): Start position
        end (int): End position
        thr_min (int, optional): Lower threshold for insert size. Defaults to 50.
        thr_max (int, optional): Upper threshold for insert size. Defaults to 1000.

    Returns:
        dict: List of low, high positions 
    """    
    
    # Open BAM file
    bam = pysam.AlignmentFile(bam_filename, 'rb')
    
    positions_exceed_min = []
    positions_exceed_max = []
    positions_rr = []
    positions_ff = []
    positions_rf = []
    positions_tx = []

    # Extract positions and insert sizes for the specified region
    for read in bam.fetch(chrom, start, end):
        if not read.is_unmapped:
            
            # Insert Size
            if abs(read.template_length) > thr_max:
                positions_exceed_max.append(read.reference_start)
                
            elif abs(read.template_length) < thr_min:
                positions_exceed_min.append(read.reference_start)
                
            # Orientation TX
            if read.reference_id != read.next_reference_id:
                positions_tx.append(read.reference_start)
            
            # Orientation RR
            elif read.is_reverse and read.mate_is_reverse:
                positions_rr.append(read.reference_start)
            
            # Orientation FF
            elif not read.is_reverse and not read.mate_is_reverse:
                positions_ff.append(read.reference_start)
            
            # Orientation RF
            elif not read.is_reverse and read.mate_is_reverse and read.template_length < 0:
                positions_rf.append(read.reference_start)
                
            elif read.is_reverse and not read.mate_is_reverse and read.template_length > 0:
                positions_rf.append(read.reference_start)
                
    positions_exceed_min = np.array(positions_exceed_min) - start
    positions_exceed_max = np.array(positions_exceed_max) - start
    positions_rr = np.array(positions_rr) - start
    positions_ff = np.array(positions_ff) - start
    positions_rf = np.array(positions_rf) - start
    positions_tx = np.array(positions_tx) - start
    
    return {'exceed_min': positions_exceed_min, 'exceed_max': positions_exceed_max, 'rr': positions_rr, 'ff': positions_ff, 'rf': positions_rf, 'tx': positions_tx}
