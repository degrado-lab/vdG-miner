import os
import sys
import glob
import gzip
import time
import errno
import signal
import argparse
import traceback

import numpy as np
import numba as nb
import prody as pr

from copy import deepcopy
from functools import wraps
from scipy.spatial.distance import cdist

from vdg_miner.database.readxml import extract_global_validation_values

"""
Updated pdb files, validation reports, and sequence clusters should be 
downloaded via the pdb ftp server:

> rsync -rlpt -v -z --delete --port=33444 
  rsync.rcsb.org::ftp_data/structures/divided/pdb/ $LOCAL_PDB_MIRROR_PATH

> rsync -rlpt -v -z --delete --include="*/" --include="*.xml.gz" --exclude="*"  
  --port=33444 rsync.rcsb.org::ftp/validation_reports/ $LOCAL_VALIDATION_PATH

> wget -O $LOCAL_CLUSTERS_PATH
  https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt

These three paths should be provided using the -e, -v, and -c arguments to 
this script, respectively.
"""


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """Raise a TimeoutError if a function takes longer than the specified time.
    
    Parameters
    ----------
    seconds : int
        Number of seconds after which to raise the TimeoutError.
    error_message : str
        Message to include in the TimeoutError.

    Returns
    -------
    decorator : function
        Decorator function that raises a TimeoutError if the wrapped 
        function takes longer than the specified time.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def read_clusters(cluster_file, job_id, num_jobs):
    """Read the sequence clusters from the RCSB cluster file.

    Parameters
    ----------
    cluster_file : str
        Path to the file containing the RCSB-curated sequence clusters.
    job_id : int
        Index for the current job, relevant for multi-job HPC runs.
    num_jobs : int
        Number of jobs, relevant for multi-job HPC runs.

    Returns
    -------
    clusters : list
        List of lists of members of clusters for the current job, stored 
        as tuples of PDB accession codes and associated polymer entity IDs.
    """
    clusters = []
    with open(cluster_file, 'r') as f:
        for line in f:
            clusters.append([])
            for member in line.split():
                mem_split = member.split('_')
                if len(mem_split[0]) == 4: # exclude non-PDB entries
                    clusters[-1].append(member.split('_'))
            if len(clusters[-1]) == 0:
                clusters.pop()
    # assign to the current job clusters corresponding to 1 / num_jobs of
    # the total number of polymer entities
    cluster_cumsum = np.cumsum([len(c) for c in clusters])
    divisor = cluster_cumsum[-1] // num_jobs
    cluster_idxs = np.argwhere(cluster_cumsum // divisor == job_id).flatten()
    return [c for i, c in enumerate(clusters) if i in cluster_idxs]


def filter_clusters(clusters, validation_dir, min_res=2.0, max_r=0.3):
    """Filter the clusters to include only those with valid resolution and R.

    Parameters
    ----------
    clusters : list
        List of lists of members of clusters, stored as tuples of PDB 
        accession codes and associated polymer entity IDs.
    validation_dir : str
        Path to directory containing xml.gz files with validation report data.
    min_res : float, optional
        Minimum resolution (in Angstroms) to permit for a PDB structure 
        (Default: 2.0).
    max_r : float, optional
        Maximum R value to permit for a PDB structure (Default: 0.3).

    Returns
    -------
    filtered_clusters : list
        List of lists of members of clusters, stored as tuples of PDB 
        accession codes and associated polymer entity IDs, that have valid 
        resolution and R.
    """
    filtered_clusters = []
    for cluster in clusters:
        filtered_clusters.append([])
        for pdb, entity in cluster:
            middle_two = pdb[1:3].lower()
            pdb_acc = pdb.lower()
            validation_file = os.path.join(validation_dir, 
                                           middle_two, pdb_acc, 
                                           pdb_acc + '_validation.xml.gz')
            try:
                resolution, r_obs = \
                    extract_global_validation_values(validation_file)
            except:
                continue
            if resolution is not None and r_obs is not None:
                if resolution <= min_res and r_obs <= max_r:
                    filtered_clusters[-1].append((pdb, entity))
        if len(filtered_clusters[-1]) == 0:
            filtered_clusters.pop()
    print(filtered_clusters)
    return filtered_clusters


def parse_compnd(ent_gz_path):
    """Parse the COMPND lines of a gzipped PDB file.
    
    Parameters
    ----------
    ent_gz_path : str
        Path to the gzipped PDB file.

    Returns
    -------
    identifier_chains_dict : dict
        Dict pairing polymer entity IDs with lists of chain IDs.
    """
    identifier_chains_dict = {}
    with gzip.open(ent_gz_path, 'rt', encoding='utf-8') as f:
        compnd_begun = False
        while True:
            line = f.readline().strip()
            if line[:6] == 'COMPND':
                compnd_begun = True
                if 'MOL_ID: ' in line:
                    idx = line.index('MOL_ID: ') + 8
                    current_id = int(line[idx:].replace(';',''))
                    identifier_chains_dict[current_id] = []
                if 'CHAIN: ' in line:
                    idx = line.index('CHAIN: ') + 7
                    chains = line[idx:].replace(';','').split(', ')
                    identifier_chains_dict[current_id] += chains
            elif compnd_begun and line[:6] != 'COMPND':
                break
    return identifier_chains_dict


def get_author_assigned_biounits(pdb_file):
    """Given a gzipped PDB file, obtain the author-assigned biounits.

    Parameters
    ----------
    pdb_file : str
        Path to gzipped PDB file for which to obtain author-assigned biounits.

    Returns
    -------
    biomols : list
        List of integer indices of biological assemblies.
    """
    biomols = []
    with gzip.open(pdb_file, 'rb') as f:
        for b_line in f:
            if b_line.startswith(b'REMARK 350'):
                line = b_line.decode('utf-8')
                if 'BIOMOLECULE:' in line:
                    biomol = int(line.strip().split('BIOMOLECULE:')[-1])
                    continue
                if 'AUTHOR DETERMINED BIOLOGICAL UNIT' in line:
                    biomols.append(biomol)
    return biomols


@timeout(5)
def get_bio(path, return_header=False):
    """Given the path to a gzipped PDB file, return its biological assemblies.

    Parameters
    ----------
    path : str
        Path to gzipped PDB file for which to return biological assemblies.\
    return_header : bool, optional
        If True, return the header of the PDB file as well.

    Returns
    -------
    bio : prody.AtomGroup or list
        ProDy AtomGroup or list of ProDy AtomGroups for the biological 
        assemblies of the structure.
    """
    with gzip.open(path, 'rt') as f:
        if return_header:
            bio, header = pr.parsePDBStream(f, biomol=True, header=True)
            if type(bio) == list:
                [b.setAnisous(None) for b in bio]
            else:
                bio.setAnisous(None)
                bio = [bio]
            return (bio, header)
        else:
            bio = pr.parsePDBStream(f, biomol=True)
            if type(bio) == list:
                [b.setAnisous(None) for b in bio]
            else:
                bio.setAnisous(None)
                bio = [bio]
            return bio
        

def assign_hydrogen_segis(biounit, out_path=None):
    """Assign segis to prepwizard-added atoms and rewrite the PDB file.

    Parameters
    ----------
    biounit : prody.AtomGroup
        ProDy AtomGroup for a biological assembly to which to assign segis.
    out_path : str, optional
        Path to which to write the PDB file with assigned segis.
    """
    struct = pr.parsePDB(biounit)
    # assign different resnums to water molecules to circumvent a bug in 
    # PrepWizard output where the same resnum is assigned to multiple waters
    resnums = struct.getResnums()
    serials = struct.getSerials()
    struct_non_HOH = struct.select('not element H and not resname HOH')
    non_HOH_segs = struct_non_HOH.getSegnames()
    non_HOH_chids = struct_non_HOH.getChids()
    non_HOH_resnums = struct_non_HOH.getResnums()
    struct_HOH = struct.select('not element H and resname HOH')
    HOH_segs = struct_HOH.getSegnames()
    HOH_chids = struct_HOH.getChids()
    HOH_resnums = struct_HOH.getResnums()
    HOH_serials = struct_HOH.getSerials()
    scrs = list(zip(non_HOH_segs, non_HOH_chids, non_HOH_resnums))
    for s, c, r, i in zip(HOH_segs, HOH_chids, HOH_resnums, HOH_serials):
        while (s, c, r) in scrs:
            r += 1
        scrs.append((s, c, r))
        resnums[serials == i] = r
    # assign segis and resnums to those atoms to which PrepWizard did not 
    # assign segis
    segis = struct.getSegnames()
    coords = struct.getCoords()
    dists = cdist(coords[segis == ''], 
                  coords[segis != ''], 
                  metric='sqeuclidean')
    argmin_dists = np.argmin(dists, axis=1)
    resnums[segis == ''] = resnums[segis != ''][argmin_dists]
    segis[segis == ''] = segis[segis != ''][argmin_dists]
    struct.setResnums(resnums)
    struct.setSegnames(segis)
    # reassign atom indices to circumvent a bug in PrepWizard output
    # for multi-segment structures
    segi_structs = [struct.select('segname ' + str(segi)).toAtomGroup() 
                    for segi in np.sort(np.unique(segis).astype(int))]
    struct = sum(segi_structs[1:], segi_structs[0])
    # write the PDB file with assigned segis
    if out_path is not None:
        pr.writePDB(out_path, struct)
    else:
        pr.writePDB(biounit, struct)


def preprocess_lines(pdb_lines, probe_lines, do_hash=True):
    """Preprocess PDB and probe lines into numpy arrays for find_neighbors.

    Parameters
    ----------
    pdb_lines : list of str
        List of ATOM lines from a PDB file.
    probe_lines : list of list of str
        List of lines, split by the character ':', from a probe file.
    do_hash : bool
        Whether or not to hash the output arrays. Default: True.
    
    Returns
    -------
    pdb_array : np.ndarray [N, ...]
        List of sliced ATOM lines from a PDB file, represented as an array.
    probe_array : np.ndarray [M, ...]
        List of sliced and rearranged lines from a probe file, represented 
        as an array.
    """
    if do_hash: # hash the arrays for speed
        # rearrange the sections of the probe lines that contain atom info 
        # to match PDB format for both the first and second atoms
        rearrangements_0 = []
        rearrangements_1 = []
        for i, probe_line in enumerate(probe_lines):
            rearrangements_0.append(hash(probe_line[3][11:15] +
                                         probe_line[3][6:11] +
                                         probe_line[3][1:6]))
            rearrangements_1.append(hash(probe_line[4][11:15] +
                                         probe_line[4][6:11] +
                                         probe_line[4][1:6]))
        # 12:26 is atom name, resname, chain, and resnum
        pdb_array = np.array([hash(line[12:26]) for line in pdb_lines], 
                              dtype=np.int64)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                               dtype=np.int64).T
    else:
        # rearrange the sections of the probe lines that contain atom info 
        # to match PDB format for both the first and second atoms
        rearrangements_0 = []
        rearrangements_1 = []
        for probe_line in probe_lines:
            rearrangements_0.append(probe_line[3][11:15] +
                                    probe_line[3][6:11] +
                                    probe_line[3][1:6])
            rearrangements_1.append(probe_line[4][11:15] +
                                    probe_line[4][6:11] +
                                    probe_line[4][1:6])
        # 12:26 is atom name, resname, chain, and resnum
        pdb_array = np.array([line[12:26] for line in pdb_lines], 
                             dtype=np.unicode_)
        probe_array = np.array([rearrangements_0, rearrangements_1], 
                               dtype=np.unicode_).T
    return pdb_array, probe_array


@nb.njit
def find_neighbors(pdb_array, probe_array, pdb_coords, probe_coords):
    """Using probe dot positions, find neighboring atoms in a PDB file.
    
    Parameters
    ----------
    pdb_array : np.ndarray [N, ...]
        List of sliced ATOM and HETATM lines from a PDB file, represented 
        as an array.
    probe_array : np.ndarray [M, ...]
        List of sliced and rearranged lines from a probe file, represented 
        as an array.
    pdb_coords : np.ndarray [N, 3]
        The coordinates of the atoms in the PDB file.
    probe_coords : np.ndarray [M, 3]
        The coordinates of the probe dots.

    Returns
    -------
    neighbors : np.ndarray [N, 2]
        The indices (starting from 0) of neighboring atoms in the PDB file.
    """
    neighbors = -100000 * np.ones((len(probe_coords), 2), dtype=np.int64)
    for i in range(len(probe_array)):
        min_distance_0 = 100000
        min_distance_1 = 100000
        for j in range(len(pdb_array)):
            if probe_array[i, 0] == pdb_array[j]:
                distance = ((pdb_coords[j] - probe_coords[i])**2).sum()
                if distance < min_distance_0:
                    min_distance_0 = distance
                    neighbors[i, 0] = j
            if probe_array[i, 1] == pdb_array[j]:
                distance = ((pdb_coords[j] - probe_coords[i])**2).sum()
                if distance < min_distance_1:
                    min_distance_1 = distance
                    neighbors[i, 1] = j
    return neighbors


def consolidate_probe_output(pdb_file, probe_file):
    """Consolidate probe output into a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file that will receive the probe output in 
        additional columns.
    probe_file : str
        Path to the probe output file to be consolidated.
    """
    pdb_lines, seg_ids, pdb_coords = [], [], []
    with open(pdb_file, 'r') as f:
        remark = f.readline()
        for line in f.read().split('\n'):
            if line.startswith('ATOM') or \
                    line.startswith('HETATM') or \
                    line.startswith('TER'):
                if len(line) > 95:
                    pdb_lines.append(line)
                else:
                    pdb_lines.append(line + ' ' * 12 * 9)
                seg_ids.append(line[72:76].strip())
                if not line.startswith('TER'):
                    pdb_coords.append([float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
                else:
                    pdb_coords.append([100000., 100000., 100000.])
    pdb_coords = np.array(pdb_coords)
    with open(probe_file, 'r') as f:
        probe_lines = [line.strip().replace('?', '2').split(':')
                       for line in f.readlines()]
    possible_cts = ['wc', 'cc', 'so', 'bo', 'wo', 'wh', 'hb']
    # wide contact, close contact, small overlap, bad overlap, worse overlap, 
    # weak hydrogen bond, hydrogen bond
    contact_types = np.array([line[2] for line in probe_lines])
    probe_coords = np.array([[float(line[8]), 
                              float(line[9]), 
                              float(line[10])] 
                              for line in probe_lines])
    pdb_array, probe_array = \
        preprocess_lines(pdb_lines, probe_lines)
    neighbors = \
        find_neighbors(pdb_array, probe_array, pdb_coords, probe_coords)
    for ct, (i, j) in zip(contact_types, neighbors):
        if i == -100000 or j == -100000:
            continue
        flag = True # flag to determine if contact has not yet been added
        for _ct in possible_cts[:possible_cts.index(ct)+1]:
            contact_str = ' ' + _ct + ' ' + str(j + 1).rjust(5)
            if contact_str in pdb_lines[i] and seg_ids[i] == seg_ids[j]:
                if _ct == ct:
                    flag = False
                    break
                start_idx = pdb_lines[i].index(contact_str)
                pdb_lines[i] = \
                    pdb_lines[i][:start_idx] + \
                    ' ' + ct + ' ' + str(j + 1).rjust(5) + \
                    pdb_lines[i][start_idx + 9:]
                flag = False
                break
        if flag: # contact has not been added, so add it
            for k in range(12):
                start_idx = 91 + k * 9
                if pdb_lines[i][start_idx:start_idx+9] != ' ' * 9:
                    continue
                pdb_lines[i] = \
                    pdb_lines[i][:start_idx] + \
                    ' ' + ct + ' ' + str(j + 1).rjust(5) + \
                    pdb_lines[i][start_idx + 9:]
                break
    with open(pdb_file, 'w') as f:
        f.write(remark)
        for line in pdb_lines:
            f.write(line + '\n')
        f.write('END   \n')
    os.remove(probe_file)


def check_status(pdb_code, pdb_outdir):
    """Check the extent to which a PDB structure has been processed.
    
    Parameters
    ----------
    pdb_code : str
        PDB code for which to check processing status.
    pdb_outdir : str
        Directory at which output from the pipeline is writted, organized 
        into subdirectories by the middle two characters of the PDB code.
        
    Returns
    -------
    status : int
        Integer denoting the processing status of the PDB structure, 
        defined as follows:
        0: PDB structure has not been processed.
        1: Biounits for the PDB structure have been written but not 
           prepped with PrepWizard.
        2: Biounits for the PDB structure have been prepped with PrepWizard 
           but not assessed with MolProbity.
        3: Biounits for the PDB structure have been assessed with MolProbity 
           and some scored well but are not consolidated with probe output.
        4: Biounits for the PDB structure have been consolidated with probe 
           output.
        5: Biounits for the PDB structure have been assessed with MolProbity 
           and all scored poorly.
    """
    middle_two = pdb_code[1:3].lower()
    pdb_subdir = os.path.join(pdb_outdir, middle_two)
    pdb_files = glob.glob(
        os.path.join(pdb_subdir, 
                     f'{pdb_code.upper()}*', 
                     f'{pdb_code.upper()}*.pdb')
    )
    molprobity_files = glob.glob(
        os.path.join(pdb_subdir, 
                     f'{pdb_code.upper()}*',
                     '*_molprobity.out')
    )
    if len(pdb_files) == 0 and len(molprobity_files) == 0:
        return 0 # no biounits written; needs write_biounits
    elif len(pdb_files) > 0:
        with open(pdb_files[0], 'rb') as f:
            for b_line in f:
                if b_line.startswith(b'ATOM'):
                    if len(b_line) > 95:
                        return 4 # biounits prepped and consolidated with probe
                    elif len(b_line) > 81:
                        return 3 # biounits prepped and scored with MolProbity; 
                                 # needs consolidation with probe
                    elif b_line.decode('utf-8')[77] == 'H':
                        return 2 # biounits prepped but no MolProbity score; 
                                 # needs MolProbity
            return 1 # no H atoms and no MolProbity score; needs PrepWizard
    else:
        return 5 # biounits scored poorly with MolProbity
    

def write_biounits(ent_gz_path, pdb_outdir, xtal_only=True, write=True):
    """For an ent.gz file, write the biological assemblies as PDB files.

    Parameters
    ----------
    ent_gz_path : str
        Path to the ent.gz file for a PDB structure to convert to a 
        biological assembly.
    pdb_outdir : str
        Directory at which to output PDB files for biological assemblies.
    xtal_only : bool, optional
        Whether to only consider X-ray crystallography structures.
    write : bool, optional
        If False, do not write the biological assemblies to PDB files.

    Returns
    -------
    bio_paths : list
        List of paths (within pdb_outdir) to PDB files containing the 
        author-assigned biological assemblies of the input PDB structure, 
        or all biounits if no author-assigned biounits are denoted.
    segs_chains = list
        List of sets of tuples, each consisting of a segment and chain 
        within each outputted biological assembly.
    """
    pdb_code = ent_gz_path.split('/')[-1][3:7]
    output_subdir = os.path.join(pdb_outdir, pdb_code[1:3])
    os.makedirs(output_subdir, exist_ok=True)
    bio_paths = []
    segs_chains = []
    bio, header = get_bio(ent_gz_path, return_header=True)
    if xtal_only and 'X-RAY' not in header['experiment']:
        return [], []
    bio_list = [int(b.getTitle().split()[-1]) for b in bio]
    author_assigned = get_author_assigned_biounits(ent_gz_path)
    if len(author_assigned) > 0:
        bio = [bio[bio_list.index(i)] for i in author_assigned]
        bio_list = author_assigned
    for i, b in enumerate(bio):
        try:
            if not b.select('protein'):
                continue
            # write biounit to PDB file
            bio_name = pdb_code.upper() + '_biounit_' + str(bio_list[i])
            os.makedirs(os.path.join(output_subdir, bio_name), exist_ok=True)
            bio_path = os.path.join(output_subdir, bio_name, 
                                    bio_name + '.pdb')
            # write the biounit to a PDB file
            b.setAltlocs(' ')
            if write:
                pr.writePDB(bio_path, b.select('not element H'))
            bio_paths.append(bio_path)
            segs_chains.append(set(zip(b.getSegnames(), b.getChids())))
        except:
            pass
    return bio_paths, segs_chains


@timeout(1800)
def prep_biounits(biounit_paths, prepwizard_path):
    """Add hydrogen to biological assemblies with Schrodinger Prepwizard.

    Parameters
    ----------
    biounit_paths : list
        List of paths to PDB files of biological assemblies to which to add
        hydrogen with Schrodinger Prepwizard.
    prepwizard_path : str
        Path to Schrodinger Prepwizard binary.

    Returns
    -------
    prepped : list
        List of bools denoting whether or not a biounit was successfully 
        prepped.
    """
    prepped = []
    cwd = os.getcwd()
    for biounit_path in biounit_paths:
        if not os.path.exists(os.path.dirname(biounit_path)):
            return # skip if already prepped and has high MolProbity score
        if os.path.exists(biounit_path[:-4] + '_molprobity.out'):
            continue # skip if already prepped and has low MolProbity score
        if not os.path.exists(biounit_path):
            continue # skip if biounit file is missing
        os.chdir(os.path.dirname(biounit_path))
        # format and execute command for Prepwizard
        tmp_path = biounit_path[:-4] + '_prepped.pdb'
        log_path = biounit_path[:-4] + '.log'
        cmd = ' '.join([prepwizard_path, biounit_path, tmp_path, 
                        '-rehtreat', '-samplewater', '-disulfides', '-mse', 
                        '-glycosylation', '-noimpref', '-use_PDB_pH', 
                        '-addOXT', '-keepfarwat'])
        # check to see if another HPC job is running prepwizard already
        if not os.path.exists(log_path):
            os.system(cmd)
        # wait until prepwizard has finished
        while not os.path.exists(tmp_path) or \
                time.time() - os.path.getmtime(tmp_path) < 5:
            time.sleep(1)
        try:
            assign_hydrogen_segis(tmp_path, biounit_path)
            prepped.append(True)
        except:
            prepped.append(False)
        os.remove(tmp_path)
        os.remove(log_path)
    os.chdir(cwd)
    return prepped


def run_molprobity(pdb_path, molprobity_path, identifier_chains_dict, 
                   cutoff=2.0):
    """Run Molprobity on a PDB file and add the score to the PDB file.

    Parameters
    ----------
    pdb_path : list
        The path to the PDB file to be scored by MolProbity.
    molprobity_path : str
        Path to Molprobity binary.
    identifier_chains_dict : dict
        Dict pairing polymer entity IDs with lists of chain IDs.
    cutoff : float
        The MolProbity score cutoff for a PDB to be considered good.

    Returns
    -------
    score : float
        The MolProbity score of the PDB file.
    """
    pdb_dirname = os.path.dirname(pdb_path)
    if not os.path.exists(pdb_dirname):
        # directory already removed for low MolProbity score (see below)
        out_path = os.path.join(
            '/'.join(pdb_path.split('/')[:2]), 
            os.path.basename(pdb_path)[:-4] + '_molprobity.out'
        )
    else:
        out_path = pdb_path[:-4] + '_molprobity.out'
        if not os.path.exists(out_path): # run molprobity
            cmd = [molprobity_path, pdb_dirname, '>', out_path]
            os.system(' '.join(cmd))
    if not os.path.exists(out_path):
        return 10000. # high score when MolProbity output is absent
    with open(out_path, 'rb') as f:
        # read last line of file to get MolProbity score
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        score_line = f.readline().decode().split(':')
    if len(score_line) > 2 and '.' in score_line[-2]:
        score = float(score_line[-2])
    else:
        score = 10000. # high score when MolProbity output is absent
    if score <= cutoff:
        # add entity IDs and MolProbity score to PDB file
        b_lines = []
        with open(pdb_path, 'rb') as f:
            for b_line in f:
                if b_line.startswith(b'ATOM') or \
                        b_line.startswith(b'HETATM'):
                    b_line = b_line.strip(b'\n')
                    chid = b_line[21:22].decode()
                    for key, val in identifier_chains_dict.items():
                        if chid in val:
                            b_line += b' ' + str(key).rjust(4).encode()
                            break
                    b_line += b' ' + str(score).rjust(5).encode() + b'\n'
                    b_lines.append(b_line)
                else:
                    b_lines.append(b_line)
        with open(pdb_path, 'wb') as f:
            for b_line in b_lines:
                f.write(b_line)
        # remove MolProbity output to indicate good-scoring PDB file
        os.remove(out_path)
    elif os.path.exists(pdb_path): 
        # save space by removing high-scoring PDB files, and keep 
        # MolProbity output for reference
        os.remove(pdb_path)
    return score


def get_score(bio_path):
    """Get the MolProbity score from a PDB file.

    Parameters
    ----------
    bio_path : str
        Path to the PDB file from which to extract the MolProbity score.
    
    Returns
    -------
    score : float
        The MolProbity score of the PDB file.
    """
    score = None
    if os.path.exists(bio_path):
        with open(bio_path, 'rb') as f:
            for b_line in f:
                if b_line.startswith(b'ATOM'):
                    if len(b_line) > 90:
                        try:
                            score = float(b_line[80:85].decode())
                        except:
                            pass
                    break
    return score


def run_probe(pdb_path, probe_path, segi='', chain='', water=False):
    """Run probe on a PDB file to find interatomic contacts.
    
    Parameters
    ----------
    pdb_path : str
        Path to the PDB file on which to run probe.
    probe_path : str
        Path to the probe binary.
    segi : str
        Segment ID for which to run probe. If None, run probe 
        for all segments.
    chain : str
        Chain ID for which to run probe. If None, run probe 
        for all chains.
    water : bool
        If True, run probe for all water molecules in the PDB file.
    """
    probe_outpath = pdb_path[:-4] + '_' + segi + '_' + chain + '.probe'
    # -U : Unformatted output
    # -SEGID : use the PDB SegID field to discriminate between residues
    # -CON : raw output in condensed format, i.e. one line per contact
    # -NOFACE : do not identify HBonds to aromatic faces
    # -WEAKH : include weak hydrogen bonds
    # -DE32 : dot density of 32 dots per square Angstrom
    # -4 : extend bond chain dot removal to 4 for H
    # -ON : single intersection (src -> targ) 
    # "WATER,SEG{},CHAIN{}" "ALL" : between water, the desired segment 
    #                               and chain, and all other atoms
    if segi == '' and chain == '':
        cmd_template = ('{} -U -SEGID -CON -NOFACE -WEAKH -DE32 -4H -ON '
                        '"ALL" "ALL" {} > {}')
        cmd = cmd_template.format(probe_path, pdb_path, probe_outpath)
    elif segi == '':
        cmd_template = ('{} -U -SEGID -CON -NOFACE -WEAKH -DE32 -4H -ON '
                        '"CHAIN{}" "ALL" {} > {}')
        cmd = cmd_template.format(probe_path, chain.rjust(2, '_'), 
                                  pdb_path, probe_outpath)
    elif chain == '':
        cmd_template = ('{} -U -SEGID -CON -NOFACE -WEAKH -DE32 -4H -ON '
                        '"SEG{}" "ALL" {} > {}')
        cmd = cmd_template.format(probe_path, segi.rjust(4, '_'), 
                                  pdb_path, probe_outpath)
    elif water:
        cmd_template = ('{} -U -SEGID -CON -NOFACE -WEAKH -DE32 -4H -ON '
                        '"WATER" "ALL" {} > {}')
        cmd = cmd_template.format(probe_path, pdb_path, probe_outpath)
    else:
        cmd_template = ('{} -U -SEGID -CON -NOFACE -WEAKH -DE32 -4H -ON '
                        '"SEG{},CHAIN{}" "ALL" {} > {}')
        cmd = cmd_template.format(probe_path, segi.rjust(4, '_'), 
                                  chain.rjust(2, '_'), pdb_path, 
                                  probe_outpath)
    os.system(cmd)
    # consolidate probe output into the PDB file
    consolidate_probe_output(pdb_path, probe_outpath)
    # with open(probe_path, 'rb') as f_in:
    #     with gzip.open(probe_path + '.gz', 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    # os.remove(probe_path)


def ent_gz_dir_to_vdg_db_files(ent_gz_dir, pdb_outdir, 
                               final_cluster_outpath, clusters, 
                               prepwizard_path, molprobity_path, 
                               probe_path, prototype='pdb{}.ent.gz'):
    """Generate vdG database files from ent.gz files and validation reports.

    Parameters
    ----------
    ent_gz_dir : str
        Path to directory containing ent.gz files from which to generate 
        input files for COMBS database generation.
    pdb_outdir : str
        Directory at which to output fully prepared PDB files for biological 
        assemblies with minimal MolProbity scores.
    final_cluster_outpath : str
        Path to output file at which to specify the biounits and chains that 
        are the lowest-Molprobity score representatives of the RCSB sequence 
        clusters.
    clusters : list
        List of list of clusters (tuples of PDB accession codes and entity 
        IDs) to prepare with prepwizard and assess with MolProbity for 
        database membership. If None, all structures in the subdirectories 
        of ent_gz_dir that match the prototype will be prepared.
    prepwizard_path : str
        Path to Schrodinger Prepwizard binary.
    molprobity_path : str
        Path to MolProbity oneline-analysis binary.
    probe_path : str
        Path to probe binary.
    prototype : str
        Prototype for the PDB filename (Default: 'pdb{}.ent.gz').
    """
    for _dir in [ent_gz_dir, pdb_outdir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    min_molprobity_clusters = []
    for cluster in clusters:
        min_molprobity_clusters.append([])
        min_cluster_score = 10000.
        for ent in cluster:
            try:
                pdb_code = ent[0].lower()
                ent_gz_path = os.path.join(ent_gz_dir, pdb_code[1:3], 
                                           prototype.format(pdb_code))
                identifier_chains_dict = parse_compnd(ent_gz_path)
                chid = identifier_chains_dict[int(ent[1])]
                if check_status(pdb_code, pdb_outdir) == 0:
                    bio_paths, segs_chains = \
                        write_biounits(ent_gz_path, pdb_outdir, xtal_only=True)
                else:
                    bio_paths, segs_chains = \
                        write_biounits(ent_gz_path, pdb_outdir, xtal_only=True, 
                                       write=False)
                if check_status(pdb_code, pdb_outdir) == 1:
                    prepped = prep_biounits(bio_paths, prepwizard_path)
                    for bio_path, is_prepped in zip(bio_paths, prepped):
                        if not is_prepped:
                            os.remove(bio_path)
                cutoff = 2.0
                scores = []
                if check_status(pdb_code, pdb_outdir) == 2:
                    for bio_path in bio_paths:
                        # check to see if the PDB file already has a score
                        score = get_score(bio_path)
                        if score is None:
                            # otherwise, run MolProbity
                            score = run_molprobity(bio_path, molprobity_path, 
                                                   identifier_chains_dict, 
                                                   cutoff)
                        scores.append(score)
                else: # status is at least 3, therefore scores are present
                    for bio_path in bio_paths:
                        scores.append(get_score(bio_path))
                if check_status(pdb_code, pdb_outdir) == 3:
                    for bio_path, sc, score in zip(bio_paths, 
                                                   segs_chains, 
                                                   scores):
                        # run probe for biounits with good MolProbity scores
                        if score is not None and score <= cutoff:
                            for seg, chain in sc:
                                run_probe(bio_path, probe_path, seg, chain)
                for bio_path, sc, score in \
                        zip(bio_paths, segs_chains, scores):
                    if score is not None and score <= min_cluster_score:
                        if score < min_cluster_score:
                            min_molprobity_clusters[-1] = []
                            min_cluster_score = score
                        biounit = bio_path.split('/')[-1][:-4]
                        for chid in identifier_chains_dict[int(ent[1])]:
                            for seg, chain in sc:
                                if chain == chid:
                                    min_molprobity_clusters[-1].append(
                                        (biounit, seg, chid)
                                    )
            except Exception:
                print('**************************************************')
                traceback.print_exc(file=sys.stdout)
                print('**************************************************')
        if len(min_molprobity_clusters[-1]) == 0:
            min_molprobity_clusters.pop()
    if len(min_molprobity_clusters):
        with open(final_cluster_outpath, 'a+') as f:
            for cluster in min_molprobity_clusters:
                for biounit, seg, chid in cluster:
                    f.write(biounit + '_' + seg + '_' + chid + ' ')
                f.write('\n')


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('-e', '--ent-gz-dir', help="Path to directory "
                      "containing ent.gz files from which to generate input "
                      "files for COMBS database generation.")
    argp.add_argument('-v', '--validation-dir', help="Path to directory "
                      "containing validation files for PDB structures "
                      "from which to extract resolution and R-free values.")
    argp.add_argument('-c', '--cluster-file', help="Path to file containing "
                      "the RCSB-curated sequence clusters in the PDB.")
    argp.add_argument('-f', '--final-cluster-outpath', default='', 
                      help="Path to output file at which to specify the "
                      "biounits and chains that are the lowest-Molprobity "
                      "score representatives of the RCSB sequence clusters.")
    argp.add_argument('-o', '--pdb-outdir', help="Directory at which to "
                      "output fully prepared PDB files for biological "
                      "assemblies with minimal MolProbity scores.")
    argp.add_argument('--prepwizard-path', required=True, 
                      help="Path to Schrodinger Prepwizard binary.")
    argp.add_argument('--molprobity-path', required=True, 
                      help="Path to MolProbity oneline-analysis binary.")
    argp.add_argument('--probe-path', required=True, 
                      help="Path to probe binary.")
    argp.add_argument('-p', '--prototype', default='pdb{}.ent.gz',
                      help="Prototype for the PDB filename (Default: "
                      "'pdb{}.ent.gz').")
    argp.add_argument('-j', '--job-index', type=int, default=0, 
                      help="Index for the current job, relevant for "
                      "multi-job HPC runs (Default: 0).")
    argp.add_argument('-n', '--num-jobs', type=int, default=1, 
                      help="Number of jobs, relevant for multi-job HPC runs "
                      "(Default: 1).")
    return argp.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # determine the sequence clusters of PDB chains to prepare, within 
    # which the elements with the best MolProbity scores will be selected
    print('Filtering clusters...')
    clusters = filter_clusters(
        read_clusters(args.cluster_file, args.job_index, args.num_jobs), 
        args.validation_dir
    )
    print('Generating database files...')
    ent_gz_dir_to_vdg_db_files(args.ent_gz_dir, args.pdb_outdir, 
                               args.final_cluster_outpath, 
                               clusters, args.prepwizard_path, 
                               args.molprobity_path, args.probe_path, 
                               args.prototype)
