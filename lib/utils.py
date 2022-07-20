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


def compute_aln_matrix(bam, chrom, start, stop, collapse_ins=True, size=100):
    """ Computes alignment matrix consisting of CIGAR integers for a given region. """
    reads = []
    for read in bam.fetch(chrom, start-5, stop+5):
        if not read.is_unmapped:
            reads.append(read)

    aln_matrix = -1 * np.ones((len(reads), size))
    aux_dict = {'split_idx' : [],
                'low_mapq_idx' : [],
                'haplotag_idx': [],
                'name' : []}

    for idx, read in enumerate(reads):
        aux_dict['name'].append(read.query_name)
        if read.has_tag('SA'):
            aux_dict['split_idx'].append(idx)
        if read.mapping_quality < 5:
            aux_dict['low_mapq_idx'].append(idx)
        if read.has_tag('HP'):
            aux_dict['haplotag_idx'].append(read.get_tag('HP'))
        else:
            aux_dict['haplotag_idx'].append(-1)
            

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
            cigararray = cigararray[mask] # Remove consecutive INS positions from CIGARARRAY
        
        if read_start < start:
            start_idx_read = start - read_start
            start_idx_read = start_idx_read
            start_idx_aln = 0
        else:
            start_idx_read = 0
            start_idx_aln = read_start - start
        if read_end > stop:
            end_idx_read = stop - read_start
            end_idx_aln = size
        else:
            end_idx_read = len(cigararray)
            end_idx_aln = read_end - start
        
        if end_idx_aln > 0 and end_idx_read > 0:
            try:
                aln_matrix[idx, start_idx_aln:end_idx_aln] = cigararray[start_idx_read:end_idx_read]
            except:
                print(start_idx_aln, end_idx_aln, start_idx_read, end_idx_read)
    
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


def compute_rep_df(rep_filename, chrom, start, stop, padding=500):
    """ Computes repeat overlap for a given region. """
    rep_df = pd.read_csv(rep_filename, index_col=0, sep='\t')
    rep_df = rep_df[(rep_df['genoName'] == chrom) & (rep_df['genoStart'] > start) & (rep_df['genoEnd'] < stop)].copy()
    rep_df['genoStart'] = rep_df['genoStart'] - start + padding
    rep_df['genoEnd'] = rep_df['genoEnd'] - start + padding
    rep_df = rep_df[(rep_df['genoEnd'] > 0) & (rep_df['genoStart'] < stop - start + (2*padding))]
    rep_df.reset_index(drop=True, inplace=True)
    return rep_df