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
                ins_idx = np.argwhere(cigararray == 1).flatten() # Find positions of INS (1) in CIGARARRAY
                diff = np.diff(ins_idx) # Compute difference between consecutive INS positions
                idx_runs = np.argwhere(diff == 1).flatten() # Find positions of consecutive INS positions
                mask = np.ones(len(cigararray), bool)
                mask[ins_idx[idx_runs]] = 0 # Deselect consecutive INS positions exept the first one
                mask[ins_idx - 1] = 0 # Deselect the position before the first INS position to avoid shifting of cigararray
                cigararray = cigararray[mask] # Remove consecutive INS positions from CIGARARRAY
            
            if read_start < start and read_end > stop:
                # Read completely covers region
                if collapse_ins:
                    start_idx_read = start - read_start
                else:
                    start_idx_read = np.argwhere(cigararray != 1).flatten()[start - read_start] # shift start by number of INS
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
                if start_idx_aln < 0:
                    print(start_idx_aln)
                
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


def compute_cov_df(bam_filename, chrom, start, stop, minq=30):
    """ Computes coverage for given region. Coverage is computed twice. Once for all reads and once for reads with
    mapping quality >= minq. """
    cov = pd.DataFrame([x.split('\t') for x in pysam.depth(bam_filename, '-r', chrom + ':' + str(start) + '-' + str(stop), '-a').split('\n')[:-1]])
    cov_minq = pd.DataFrame([x.split('\t') for x in pysam.depth(bam_filename, '-r', chrom + ':' + str(start) + '-' + str(stop), '-a', '-Q', str(minq)).split('\n')[:-1]])
    cov[1] = cov[1].astype(int)
    cov[2] = cov[2].astype(int)
    cov[1] = cov[1] - cov.loc[0, 1]
    cov_minq[1] = cov_minq[1].astype(int)
    cov_minq[2] = cov_minq[2].astype(int)
    cov_minq[1] = cov_minq[1] - cov_minq.loc[0, 1]

    return cov, cov_minq


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
    pos = str(pos)
    newstr = ''
    for i in range(len(pos)):
        newstr = pos[::-1][i] + newstr
        if i % 3 == 2:
            newstr = ',' + newstr
    
    if newstr[0] == ',':
        newstr = newstr[1:]
    return newstr


def compute_baseline_cov(bam_filename: str, chrom: str, n: int=1000, s: int=1000) -> float:
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
        
    baseline_coverage_mean = []
    for i in range(n):
        start = np.random.randint(0, bam.lengths[chrom_idx])
        stop = start + s
        
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
    print(positions_rr)
    print(positions_ff)
    print(positions_rf)
    positions_tx = np.array(positions_tx) - start
    
    return {'exceed_min': positions_exceed_min, 'exceed_max': positions_exceed_max, 'rr': positions_rr, 'ff': positions_ff, 'rf': positions_rf, 'tx': positions_tx}