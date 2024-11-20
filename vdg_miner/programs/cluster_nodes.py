import os
import sys
import glob
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from itertools import permutations
from collections import defaultdict

import numpy as np
import prody as pr

from Bio import pairwise2
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from vdg_miner.constants import aas, three_to_one


def rebase_path(child_path, old_parent, new_parent):
    """Given a child path and a parent, switch the child to a new parent.

    Parameters
    ----------
    child_path : str
        Path to a folder or file.
    old_parent : str
        Path to a folder containing the child_path, perhaps many levels up in 
        the directory hierarchy.
    new_parent : str
        Path to a new folder in which to place the child.
    """
    # Convert the paths to Path objects
    child_path = Path(child_path)
    old_parent = Path(old_parent)
    new_parent = Path(new_parent)
    
    # Get the relative path of the child with respect to the old parent
    relative_path = child_path.relative_to(old_parent)
    
    # Combine the new parent with the relative path
    new_path = new_parent / relative_path
    
    # Return the new path as a string
    return str(new_path)


def permute_with_symmetry(symmetry_classes):
    """Get all possible permutation arrays that preserve symmetry classes.
    
    Parameters
    ----------
    symmetry_classes : list
        List of integers representing the symmetry classes of the elements.
        
    Returns
    -------
    valid_permutations : list
        List of valid permutations that preserve symmetry classes.
    """
    elements = list(range(len(symmetry_classes)))
    # Get all possible permutations
    all_permutations = permutations(elements)
    # Filter permutations based on symmetry classes
    valid_permutations = []
    for perm in all_permutations:
        is_valid = True
        for i, p in enumerate(perm):
            # Compare the symmetry classes of elements in the original list
            # and the permuted list at the same positions
            if symmetry_classes[elements.index(p)] != symmetry_classes[i]:
                is_valid = False
                break
        if is_valid:
            valid_permutations.append(np.array(perm))
    return valid_permutations

def greedy(adj_mat, min_cluster_size=2):
    """Greedy clustering algorithm based on an adjacency matrix.
        
        Takes an adjacency matrix as input.
        All values of adj_mat are 1 or 0:  1 if <= to cutoff, 
        0 if > cutoff.

        The diagonal of adj_mat should be 0.

    Parameters
    ----------
    adj_mat : scipy.sparse.csr_matrix
        Adjacency matrix of the graph.
    min_cluster_size : int, optional
        Minimum size of a cluster, by default 2.

    Returns
    -------
    all_mems : list
        List of arrays of cluster members.
    cents : list
        List of cluster centers.
    """
    if not isinstance(adj_mat, csr_matrix):
        try:
            adj_mat = csr_matrix(adj_mat)
        except:
            print('adj_mat distance matrix must be scipy csr_matrix '
                  '(or able to convert to one)')
            return

    assert adj_mat.shape[0] == adj_mat.shape[1], \
        'Distance matrix is not square.'

    all_mems = []
    cents = []
    indices = np.arange(adj_mat.shape[0])

    try:
        while adj_mat.shape[0] > 0:

            cent = adj_mat.sum(axis=1).argmax()
            row = adj_mat.getrow(cent)
            tf = ~row.toarray().astype(bool)[0]
            mems = indices[~tf]

            if len(mems) < min_cluster_size:
                [cents.append(i) for i in indices]
                [all_mems.append(np.array([i])) for i in indices]
                break

            cents.append(indices[cent])
            all_mems.append(mems)

            indices = indices[tf]
            adj_mat = adj_mat[tf][:, tf]
    except KeyboardInterrupt:
        pass

    return all_mems, cents


def kabsch(X, Y, w=None):
    """Rotate and translate X into Y to minimize the MSD between the two.
       
       Implements the SVD method by Kabsch et al. (Acta Crystallogr. 1976, 
       A32, 922).

    Parameters
    ----------
    X : np.array [M x N x 3]
        Array of M sets of mobile coordinates (N x 3) to be transformed by a 
        proper rotation to minimize mean squared displacement (MSD) from Y.
    Y : np.array [M x N x 3]
        Array of M sets of stationary coordinates relative to which to 
        transform X.
    W : np.array [N], optional
        Vector of weights for fitting.

    Returns
    -------
    R : np.array [M x 3 x 3]
        Proper rotation matrices required to transform each set of coordinates 
        in X such that its MSD with the corresponding coordinates in Y is 
        minimized.
    t : np.array [M x 3]
        Translation matrix required to transform X such that its MSD with Y 
        is minimized.
    msd : np.array [M]
        Mean squared displacement after alignment for each pair of coordinates.
    """
    N = X.shape[1]
    if w is None:
        w = np.ones((1, N, 1)) / N
    else:
        w = w.reshape((1, -1, 1)) / w.sum()
    Xw, Yw = X * w, Y * w
    # compute R using the Kabsch algorithm
    Xbar, Ybar = np.mean(Xw, axis=1, keepdims=True), \
                 np.mean(Yw, axis=1, keepdims=True)
    Xc, Yc = Xw - Xbar, Yw - Ybar
    H = np.matmul(np.transpose(Xc, (0, 2, 1)), Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.matmul(U, Vt)))
    D = np.zeros((X.shape[0], 3, 3))
    D[:, 0, 0] = 1.
    D[:, 1, 1] = 1.
    D[:, 2, 2] = d
    R = np.matmul(U, np.matmul(D, Vt))
    t = (Ybar - np.matmul(Xbar, R)).reshape((-1, 3))
    # compute MSD from aligned coordinates XR
    XRmY = np.matmul(Xc, R) - Yc
    msd = np.sum(XRmY ** 2, axis=(1, 2))
    return R, t, msd


def cluster_structures(directory, starting_dir, outdir, target_depth, 
                       cg='gn', cutoff=1.0, idxs=None, symmetry_classes=None):
    """Greedily cluster the structures in a directory based on RMSD.

    Parameters
    ----------
    directory : str
        The directory containing the structures to cluster.
    starting_dir : str
        The top-level directory of the hierarchy.
    outdir : str
        The new top-level directory at which the clustered structures will be 
        output.
    target_depth : int
        The depth below which directories should be clustered.
    cg : str, optional
        The chemical group at the center of each structure, by default 'gn'.
    cutoff : float, optional
        The RMSD cutoff for greedy clustering, by default 1.0.
    idxs : list, optional
        Indices of CG atoms on which to cluster, by default None. If None, 
        all CG atoms are used.
    symmetry_classes : list, optional
        Integers representing the symmetry classes of the CG atoms on which
        clustering is to be performed. If provided, should have the same
        length as idxs. If not provided, the atoms are assumed to be
        symmetrically inequivalent.
    """
    def get_depth(path):
        return path[len(starting_dir):].count(os.sep)
    current_depth = get_depth(directory)
    if current_depth < target_depth or \
            'seqdist' in os.path.basename(directory) or \
            '_' not in os.path.basename(directory):
        # current directory is not an ABPLE-labeled directory
        return
    subdirs = directory.split('/')
    node_aas = []
    node_aa_counts = {aa : 0 for aa in aas}
    for i, subdir in enumerate(subdirs):
        flag = False
        for aa in aas:
            if subdir[:3] == aa:
                node_aas.append((aa, node_aa_counts[aa]))
                node_aa_counts[aa] += 1
                flag = True # the most recent subdir is an amino acid
                break
        '''
        if i == len(subdirs) - 2 and not flag:
            # the second-to-last parent subdirectory is not an amino acid
            return
        '''
    pdbs = [os.path.realpath(f) for f in 
            glob.glob(directory + '/**/*.pdb', recursive=True)]
    #         if 'clusters' not in f]
    structs = [pr.parsePDB(pdb) for pdb in pdbs]
    if idxs is None:
        occs0 = structs[0].getOccupancies()
        idxs = list(range((occs0 >= 3.).sum()))
    if symmetry_classes is None:
        symmetry_classes = list(range(len(idxs)))
    assert len(symmetry_classes) == len(idxs), \
        'Length of symmetry_classes must match length of idxs.'
    perms = permute_with_symmetry(symmetry_classes)
    coords = []
    for i in range(len(pdbs)):
        occs = structs[i].getOccupancies()
        names = structs[i].getNames()
        resnames = structs[i].getResnames()
        all_coords = structs[i].getCoords()
        cg_idxs = np.array(
            [np.argwhere(occs == 3. + 0.1 * idx)[0][0] for idx in idxs]
        )
        coords_to_add = []
        for perm in perms:
            perm_coords = []
            perm_coords.append(
                all_coords[cg_idxs[perm]]
            )
            for j in range(len(node_aas)):
                for name in ['N', 'CA', 'C']: #, O]:
                    mask = np.logical_and.reduce((occs > 1.,
                                                  names == name,
                                                  resnames == node_aas[j][0]))
                    perm_coords.append(
                        all_coords[mask][node_aas[j][1]:node_aas[j][1]+1]
                    )
            coords_to_add.append(np.vstack(perm_coords))
        coords += coords_to_add
    coords = np.array(coords).reshape((len(pdbs), len(perms), -1, 3))
    coords = coords.transpose((1, 0, 2, 3))
    M = coords.shape[1]
    if M == 1:
        return
    N = len(idxs) + \
        len(node_aas) * 3 # number of atoms in CG plus N,CA,C for each aa
    # define equal weights for the CG and the mainchain atoms of 
    # interacting residues to be used in the Kabsch alignments
    weights = np.zeros(N)
    weights[:len(idxs)] = 0.5 / len(idxs)
    weights[len(idxs):] = 0.5 / (len(node_aas) * 3)
    assert np.abs(weights.sum() - 1.) < 1e-8
    # find minimal-RMSD alignments between all pairs of structures
    triu_indices = np.triu_indices(M, 1)
    L = len(triu_indices[0])
    R, t, msd, best_perms = np.zeros((L, 3, 3)), np.zeros((L, 3)), \
                            10000. * np.ones(L), np.zeros(L, dtype=int)
    for i in range(coords.shape[0]):
        # iterate over atom permutations to find the best alignments
        _R, _t, _msd = kabsch(coords[i][triu_indices[0]], 
                              coords[0][triu_indices[1]], 
                              weights)
        R[_msd < msd] = _R[_msd < msd]
        t[_msd < msd] = _t[_msd < msd]
        best_perms[_msd < msd] = i
        msd[_msd < msd] = _msd[_msd < msd]
    A = np.eye(M, dtype=int)
    A[triu_indices] = (msd <= cutoff ** 2).astype(int)
    A = A + A.T
    all_mems, cents = greedy(A)
    if set([len(cluster) for cluster in all_mems]) == {1}:
        return
    cl_directory = rebase_path(directory, starting_dir, outdir)
    cluster_dirname = os.path.join(cl_directory, 'clusters')
    os.makedirs(cluster_dirname, exist_ok=True)
    cluster_num = 1
    for cluster, cent in zip(all_mems, cents):
        assert cent in cluster, f'Centroid {cent} not in cluster {cluster}.'
        if len(cluster) < 2:
            continue
        # cluster_sorted = [cent] + [el for el in cluster if el != cent]
        cluster_subdirname = \
            os.path.join(cluster_dirname, 'cluster_{}'.format(cluster_num))
        os.makedirs(cluster_subdirname, exist_ok=True)
        seqs = []
        for el in cluster: # cluster_sorted:
            pdb_name = pdbs[el]
            cl_struct = structs[el]
            # ensure the matching environments are not homologous
            seq = ''.join([three_to_one[resname] for resname in 
                           cl_struct.getResnames() if resname in aas])
            homologous = False
            for _seq in seqs:
                aligned_seq1, aligned_seq2, score, start, end = \
                    pairwise2.align.globalxx(seq, _seq)[0]
                matches = \
                    sum(1 for a, b in zip(aligned_seq1, aligned_seq2) 
                        if a == b and a != '-' and b != '-')
                if (matches / min(len(seq), len(_seq))) > 0.7:
                    # greater than 70% identity with another sequence 
                    # in the cluster, so do not include this site
                    homologous = True
                    break
            if el != cent and homologous:
                continue
            else:
                seqs.append(seq)
            # create the environment PDB file
            if el == cent:
                pdb_name = pdb_name[:-4] + '_centroid.pdb'
            else:
                el_mobile = np.logical_and(triu_indices[0] == el,
                                           triu_indices[1] == cent)
                cent_mobile = np.logical_and(triu_indices[0] == cent,
                                             triu_indices[1] == el)
                if np.any(el_mobile):
                    # non-centroid was mobile in the Kabsch alignment
                    idx = np.argwhere(el_mobile).flatten()[0]
                    _R = R[idx]
                    _t = t[idx]
                    rmsd = np.sqrt(msd[idx])
                    # non-centroid was mobile, so it was permuted
                    # cent_coords = coords[0][cent]
                    # el_coords = coords[best_perms[idx]][el]
                else:
                    # centroid was mobile in the Kabsch alignment
                    idx = np.argwhere(cent_mobile).flatten()[0]
                    _R = R[idx].T
                    _t = -np.dot(t[idx], _R)
                    rmsd = np.sqrt(msd[idx])
                    # centroid was mobile, so el was permuted
                    # cent_coords = coords[best_perms[idx]][cent]
                    # el_coords = coords[0][el]
                '''
                el_transformed = np.dot(el_coords, _R) + _t
                test_msd = ((cent_coords - el_transformed) ** 2).sum()
                test_rmsd = np.sqrt(test_msd)
                assert np.abs(test_rmsd - rmsd) < 1e-8
                '''
                cl_struct.setCoords(
                    np.dot(cl_struct.getCoords(), _R) + _t
                )
                print('RMSD =', rmsd)
            pr.writePDB(os.path.join(cluster_subdirname, 
                                     os.path.basename(pdb_name)), cl_struct)
        subdir_files = os.listdir(cluster_subdirname)
        n_files = len(subdir_files)
        if n_files == 1:
            os.remove(os.path.join(cluster_subdirname, subdir_files[0]))
            os.rmdir(cluster_subdirname)
        else:
            os.rename(cluster_subdirname, 
                      cluster_subdirname + f"_size_{n_files}")
        cluster_num += 1
    if len(os.listdir(cluster_dirname)) == 0:
        os.rmdir(cluster_dirname)


def cluster_structures_at_depth(starting_dir, outdir, target_depth, 
                                cutoff=1.0, idxs=None, symmetry_classes=None, 
                                n_threads=8):
    """
    Traverses the directory tree starting from `starting_dir` and clusters 
    the structures in each sub-tree at a specified depth.

    Parameters
    ----------
    starting_dir : str
        The root directory from which to start the traversal.
    outdir : str
        The directory at which to output clustered PDB files.
    target_depth : int
        The depth below which directories should be clustered.
    cutoff : float, optional
        The RMSD cutoff for greedy clustering, by default 1.0.
    idxs : list, optional
        Indices of CG atoms on which to cluster, by default None. If None, 
        all CG atoms are used.
    symmetry_classes : list, optional
        Integers representing the symmetry classes of the CG atoms on which
        clustering is to be performed. If provided, should have the same
        length as idxs. If not provided, the atoms are assumed to be
        symmetrically inequivalent.
    n_threads : int, optional
        The number of threads on which to run the process.
    """
    cg = starting_dir.split('/')[-1].split('_')[0]
    os.makedirs(outdir, exist_ok=True)
    directories, _, _ = zip(*os.walk(starting_dir, topdown=False))
    func = partial(cluster_structures, 
                   starting_dir=starting_dir, outdir=outdir, 
                   target_depth=target_depth, cg=cg, 
                   cutoff=cutoff, idxs=idxs, 
                   symmetry_classes=symmetry_classes)
    with Pool(n_threads) as p:
        p.map(func, directories)
    '''
    for root, dirs, files in :
            #     root.split('_')[-1] != '1':
            # Cluster the structures in the current directory
            cluster_structures(root, starting_dir, outdir, cg=cg, 
                               cutoff=cutoff, idxs=idxs, 
                               symmetry_classes=symmetry_classes)
    '''


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster structures in a '
                                     'directory tree based on RMSD.')
    parser.add_argument('starting_dir', type=os.path.realpath, 
                        help='The root directory from which to start the '
                        'traversal.')
    parser.add_argument('outdir', type=os.path.realpath, 
                        help='The output directory at which the post-'
                        'clustering PDB files will be written in a directory '
                        'tree with the same structure as starting_dir.')
    parser.add_argument('target_depth', type=int, help='The depth below which '
                        'directories should be clustered.')
    parser.add_argument('cutoff', type=float, help='The RMSD cutoff for '
                        'greedy clustering.')
    parser.add_argument('-i', '--idxs', nargs='+', type=int, 
                        help='Indices of CG atoms on which to cluster.')
    parser.add_argument('-s', '--symmetry-classes', nargs='+', type=int,
                        help='Integers representing the symmetry classes of '
                             'the CG atoms on which clustering is to be '
                             'performed. If provided, should have the same '
                             'length as idxs. If not provided, the atoms '
                             'are assumed to be symmetrically inequivalent.')
    parser.add_argument('-t', '--threads', type=int, default=8, 
                        help="Number of threads on which to run the process.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cluster_structures_at_depth(args.starting_dir, args.outdir, 
                                args.target_depth, args.cutoff, 
                                args.idxs, args.symmetry_classes, 
                                args.threads)