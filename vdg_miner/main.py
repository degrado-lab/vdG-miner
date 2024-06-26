import os
import sys
import gzip
import errno
import signal
import shutil
import pickle
import argparse
import traceback

import numpy as np
import prody as pr

from copy import deepcopy
from functools import wraps
# from seq3Di import calc_3Di
from scipy.spatial.distance import cdist

"""
Updated pdb files and validation reports should be downloaded via the 
pdb ftp server:

> rsync -rlpt -v -z --delete --port=33444 
  rsync.rcsb.org::ftp_data/structures/divided/pdb/ $LOCAL_PDB_MIRROR_PATH

> rsync -rlpt -v -z --delete --include="*/" --include="*.xml.gz" --exclude="*"  
  --port=33444 rsync.rcsb.org::ftp/validation_reports/ $LOCAL_VALIDATION_PATH

These two paths should be provided using the -e and -v arguments to this 
script, respectively.
"""

resnames_aa_20 = ['CYS', 'ASP', 'SER', 'GLN', 'LYS',
                  'ILE', 'PRO', 'THR', 'PHE', 'ASN',
                  'GLY', 'HIS', 'LEU', 'ARG', 'TRP',
                  'ALA', 'VAL', 'GLU', 'TYR', 'MET',
                  'MSE']
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
non_prot_sel = 'not resname ' + ' and not resname '.join(resnames_aa_20)

LETTERS = 'ACDEFGHIKLMNPQRSTVWYZ'
invalid_state = 'X'

script_dir = os.path.dirname(os.path.abspath(__file__))
encoder_path = os.path.join(script_dir, "encoder.pkl")
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
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


def get_segs_chains_resnums(atomgroup, selection, resnames=False):
    """Get a set containing tuples of segments, chains, and resnums.

    Parameters
    ----------
    atomgroup : prody.atomic.atomgroup.AtomGroup
        ProDy AtomGroup for a protein structure.
    selection : str
        String specifying subset of atoms in the ProDy selection algebra.
    resnames : bool
        If True, return residue names in each tuple as well.

    Returns
    -------
    segs_chains_resnums : set
        Set containing tuples of segments, chains, and resnums for 
        each residue that matches the selection string.
    """
    sel = atomgroup.select(selection)
    if sel is None:
        return set()
    if resnames:
        return set(zip(sel.getSegnames(), sel.getChids(), sel.getResnums(), 
                       sel.getResnames()))
    else:
        return set(zip(sel.getSegnames(), sel.getChids(), sel.getResnums()))


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
    _str = 'AUTHOR DETERMINED BIOLOGICAL UNIT'
    biomols = []
    with gzip.open(pdb_file, 'rt') as infile:
        for line in infile:
            if line[:10] == 'REMARK 350':
                break
        for line in infile:
            if line[:10] != 'REMARK 350':
                break
            if 'BIOMOLECULE:' in line:
                biomol = int(line.strip().split('BIOMOLECULE:')[-1])
                continue
            if _str in line:
                biomols.append(biomol)
    return biomols


@timeout(5)
def get_bio(path):
    """Given the path to a gzipped PDB file, return its biological assemblies.

    Parameters
    ----------
    path : str
        Path to gzipped PDB file for which to return biological assemblies.

    Returns
    -------
    bio : prody.AtomGroup or list
        ProDy AtomGroup or list of ProDy AtomGroups for the biological 
        assemblies of the structure.
    """
    with gzip.open(path, 'rt') as f:
        return pr.parsePDBStream(f, biomol=True)


def write_biounits(ent_gz_paths, pdb_tmp_dir, max_ligands=None, write=True):
    """For a list of ent.gz files, write the author-assigned biounits to PDB.

    Parameters
    ----------
    ent_gz_paths : list
        List of paths to ent.gz files for PDB structures.
    pdb_tmp_dir : str
        Temporary directory at which to output unzipped ent files.
    max_ligands : int, optional
        Maximum number of heteroatom (i.e. non-protein, non-nucleic, and 
        non-water) residues to permit in a biological assembly.
    write : bool, optional
        If False, do not write the biological assemblies to PDB files.

    Returns
    -------
    bio_paths : list
        List of paths (within pdb_tmp_dir) to biounit PDB files.
    """
    if pdb_tmp_dir[-1] != '/':
        pdb_tmp_dir += '/'
    bio_paths = []
    for i, path in enumerate(ent_gz_paths):
        try:
            bio = get_bio(path)
            pdb_code = path.split('/')[-1][:7]
            if type(bio) != list:
                bio = [bio]
            # bio_list = [k + 1 for k in range(len(bio))]
            bio_list = [int(b.getTitle().split()[-1]) for b in bio]
            author_assigned = get_author_assigned_biounits(path)
            if len(author_assigned) > 0:
                # bio_list = [int(b.getTitle().split()[-1]) for b in bio]
                bio = [bio[bio_list.index(i)] for i in author_assigned]
                bio_list = author_assigned
            for i, b in enumerate(bio):
                water_sel = 'not water' # or water within 4 of protein'
                bio[i] = b.select(water_sel).toAtomGroup()
            n_near = [len(get_segs_chains_resnums(b, 
                      non_prot_sel + ' within 4 of protein')) 
                      for b in bio]
            if type(max_ligands) is int:
                n_ligands = \
                    [len(get_segs_chains_resnums(b, 'not water hetero')) 
                     for b in bio]
                bio = [b for b, nl, nn in zip(bio, n_ligands, n_near) 
                       if nl < max_ligands and nn > 0]
            else:
                bio = [b for b, nn in zip(bio, n_near) if nn > 0] 
            for i, b in enumerate(bio):
                if not b.select('protein'):
                    continue
                b = b.select('protein or same residue as within 4 of protein')
                
                '''
                # compute 3Di states
                states = calc_3Di(encoder, bio[i])
                ca_coords = np.vstack(
                        [res.select('name CA').getCoords() for i, res in 
                         enumerate(bio[i].iterResidues()) if states[i] != 20])
                dists = cdist(ca_coords, ca_coords, metric='sqeuclidean')
                nbrs = np.argsort(dists, axis=1)[:, :50]
                nbr_states = states[states < 20][nbrs]
                nbr_states_all = np.full((len(states), 50), 20, dtype=np.uint8)
                nbr_states_all[states < 20][:, :nbr_states.shape[1]] = \
                    nbr_states
                # seq = ''.join([LETTERS[state] if state != -1 
                #                else invalid_state for state in states])
                '''

                # determine one-letter sequence
                resnames = [res.getResname() for res in bio[i].iterResidues()]
                seq = ''.join([three_to_one[rn] if rn in three_to_one.keys() 
                               else 'X' for rn in resnames])
                
                # write biounit to PDB
                bio_path = pdb_tmp_dir + pdb_code[3:].upper() + '_biounit_' + \
                           str(bio_list[i]) + '.pdb'
                if write:
                    assert False, "SHOULDN'T BE HERE!"
                    pr.writePDB(bio_path, bio[i])
                    # np.save(bio_path[:-4] + '.npy', nbr_states_all)
                    with open(bio_path[:-4] + '.seq', 'w') as f:
                        f.write(seq)
                bio_paths.append(bio_path)
        except Exception:
            print('**************************************************')
            traceback.print_exc(file=sys.stdout)
    return bio_paths


@timeout(1800)
def reduce_pdb(pdb_path, reduce_path, hetdict_path=None):
    """Add hydrogens to a PDB file using the Reduce program.

    Parameters
    ----------
    pdb_path : list
        The path to the PDB file to be reduced.
    reduce_path : str
        Path to reduce binary.
    hetdict_path : str
        Path to het_dict specifying for the Reduce program how ligands 
        should be protonated.
    """
    with open(pdb_path, "r") as f:
        if 'USER  MOD reduce' in f.readline():
            return
    to_reduce_path = pdb_path + '_to_reduce'
    shutil.move(pdb_path, to_reduce_path)
    trimreduce_path = pdb_path + '_trimreduce'
    cmd = [reduce_path, '-TRIM', to_reduce_path, '>', 
           trimreduce_path, ';', reduce_path]
    if hetdict_path is not None:
        cmd += ['-DB', hetdict_path]
    cmd += ['-BUILD', trimreduce_path, '>', 
            pdb_path, ';', 'rm', to_reduce_path, 
            ';', 'rm', trimreduce_path]
    os.system(' '.join(cmd))


def run_molprobity(pdb_path):
    """Run Molprobity on a PDB file and gzip the results.

    Parameters
    ----------
    pdb_path : list
        The path to the PDB file to be reduced.
    """
    pdb_prefix = pdb_path[:-4]
    out_path = pdb_prefix + '_molprobity.out'
    probe_path = pdb_prefix + '_probe.txt'
    # if os.path.exists(out_path):
    #     return
    cmd = ['phenix.molprobity', pdb_path, ';',  
           'mv', 'molprobity.out', out_path, ';', 
           'rm', 'molprobity_coot.py']
    os.system(' '.join(cmd))
    if not os.path.exists(out_path):
        return
    with open(out_path, 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        score = float(f.readline().decode().split()[-1])
    if score <= 2.0:
        cmd = ['mv', 'molprobity_probe.txt', probe_path, ';', 
               'gzip', probe_path]
    else:
        cmd = ['rm', 'molprobity_probe.txt']
    os.system(' '.join(cmd))



def ent_gz_dir_to_vdg_db_files(ent_gz_dir, pdb_tmp_dir, 
                               reduce_path, max_ligands=25, 
                               pdb_het_dict=None, retry=False):
    """Generate input files for COMBS database generation from ent.gz files.

    Parameters
    ----------
    ent_gz_dir : str
        Path to directory containing ent.gz files from which to generate input 
        files for COMBS database generation.
    validation_dir : str
        Path to directory containing xml.gz files with validation report 
        data for all PDB structures.
    prody_pkl_outdir : str
        Path to directory at which to output pickled ProDy files.
    pdb_tmp_dir : str
        Temporary directory at which to output unzipped ent files.
    reduce_path : str
        Path to reduce binary.
    max_ligands : int
        Maximum number of heteroatom (i.e. non-protein, non-nucleic, and 
        non-water) residues to permit in a biological assembly.
    pdb_het_dict : str
        Path to het_dict specifying for the Reduce program how ligands 
        should be protonated.
    retry : bool
        Run as if the code has already been run but did not complete.
    """
    for _dir in [ent_gz_dir, pdb_tmp_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    ent_gz_paths = [os.path.join(ent_gz_dir, path) 
                    for path in os.listdir(ent_gz_dir)]
    bio_paths = write_biounits(ent_gz_paths, pdb_tmp_dir, 
                               max_ligands, write=(not retry))
    cwd = os.getcwd()
    if not retry:
        for bio_path in bio_paths:
            os.chdir(os.path.dirname(bio_path))
            reduce_pdb(bio_path, reduce_path, pdb_het_dict)
            run_molprobity(bio_path)
    os.chdir(cwd)


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('-e', '--ent-gz-dir', help="Path to directory "
                      "containing ent.gz files from which to generate input "
                      "files for COMBS database generation.")
    argp.add_argument('-t', '--pdb-tmp-dir', 
                      default='/wynton/scratch/rian.kormos/tmp_pdbs/', 
                      help="Temporary directory at which to output unzipped "
                      "ent files.")
    argp.add_argument("--reduce-path", help="Path to reduce binary.")
    argp.add_argument('-m', '--max-ligands', type=int, default=25, 
                      help="Maximum number of heteroatom (i.e. non-protein, "
                      "non-nucleic, and non-water) residues to permit in a "
                      "biological assembly.")
    argp.add_argument('-d', '--pdb-het-dict', help="Path to het_dict "
                      "specifying for the Reduce program how ligands "
                      "should be protonated (optional).")
    argp.add_argument('--retry', action='store_true', 
                      help="Run as if the code has already been run but "
                      "did not complete (i.e. finish generating the files "
                      "that did not generate in an earlier run.")
    argp.add_argument('--reduce-only', action='store_true', 
                      help=("If True, only reduce the PDB files in the "
                            "temporary directory."))
    argp.add_argument('--molprobity-only', action='store_true', 
                      help=("If True, only run Molprobity on the PDB files "
                            "in the temporary directory."))
    return argp.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _reduce = '/wynton/home/degradolab/rkormos/reduce/reduce_src/reduce'
    _het_dict = ('/wynton/home/degradolab/rkormos/reduce/'
                 'reduce_wwPDB_het_dict.txt')
    if args.reduce_path is None:
        args.reduce_path = _reduce
    if args.pdb_het_dict is None:
        args.pdb_het_dict = _het_dict
    if not args.reduce_only and not args.molprobity_only:
        ent_gz_dir_to_vdg_db_files(args.ent_gz_dir, args.pdb_tmp_dir, 
                                   args.reduce_path, args.max_ligands, 
                                   args.pdb_het_dict, args.retry)
    elif args.reduce_only:
        for filepath in os.listdir(args.pdb_tmp_dir):
            if filepath[-4:] == '.pdb':
                pdb_path = os.path.join(args.pdb_tmp_dir, filepath)
                reduce_pdb(pdb_path, args.reduce_path, args.pdb_het_dict)
    elif args.molprobity_only:
        for filepath in os.listdir(args.pdb_tmp_dir):
            if filepath[-4:] == '.pdb':
                pdb_path = os.path.join(args.pdb_tmp_dir, filepath)
                run_molprobity(pdb_path)
                
