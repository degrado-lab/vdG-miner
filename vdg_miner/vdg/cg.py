
import os
import sys
import gzip
import subprocess

from openbabel import openbabel as ob

from vdg_miner.constants import b_aas, elements

# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
    
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def find_line_in_file_grep(file_path, target_string):
    """Use grep to find a line in a file containing a target string.

    Parameters
    ----------
    file_path : str
        Path to file.
    target_string : str
        Target string to search for in the file.
    
    Returns
    -------
    line : int
        Line in the file containing the target string.
    """
    line_str = subprocess.run(['grep', '-n', target_string, file_path], 
                              stdout=subprocess.PIPE, 
                              text=True).stdout.split(':')[0]
    if len(line_str):
        return int(line_str)
    else:
        return -1

def find_cg_matches(smarts_pattern, pdb_path, pdb_cluster_file, 
                    cluster_min_scores, cg_match_dicts, 
                    include_water=False):
    """
    Find CGs matching a SMARTS pattern in a PDB file.
    
    Parameters
    ----------
    smarts_pattern : str
        SMARTS pattern.
    pdb_path : str
        Path to directory containing PDB files organized in subdirectories 
        by the middle two characters of the PDB ID.
    pdb_cluster_file : str
        Path to file containing the RCSB-curated sequence clusters in the PDB.
    cluster_min_scores : dict
        Dictionary with sorted, non-redundant tuples of entity ID clusters 
        (from pdb_cluster_file) of protein chains that contact the CG matches 
        as keys and the minimum MolProbity score of a structure containing a 
        CG match for which the contacting protein chains have entity IDs 
        that match the key. This dict will be updated if the pdb_path is 
        the new minimum for a given key.
    cg_match_dicts : dict
        Dictionary with sorted, non-redundant tuples of entity ID clusters 
        (from pdb_cluster_file) of protein chains that contact the CG matches 
        as keys and, as values, dictionaries of matching CGs in ligands, with 
        tuples of (struct_name, seg, chain, resnum, resname) for the ligand as 
        keys and, as values, lists that contain the list of atom names, the 
        MolProbity score, and the set of cluster numbers of the contacting 
        entities for each match to the CG. Used for non-protein CGs. This dict 
        will be updated if the pdb_path is the new minimum for a given key.
    include_water : bool, optional
        Whether to include water molecules in the search. Default is False.
    """
    biounit = pdb_path.split('/')[-1][:-4]
    pdb_code = biounit[:4].upper()

    # Initialize Open Babel conversion object
    obConversion = ob.OBConversion()
    obConversion.SetInFormat('pdb')

    # Initialize Open Babel SMARTS matcher
    smarts = ob.OBSmartsPattern()
    smarts.Init(smarts_pattern)

    # Extract element names from the SMARTS pattern
    smarts_elements = []
    for i in range(smarts.NumAtoms()):
        smarts_elements.append(elements[smarts.GetAtomicNum(i)])

    # Read PDB file and extract ligands as blocks, then search them for CG
    ligands = {} # PDB blocks for each ligand
    has_element = {} # lists of bools determining whether each ligand has the 
                     # necessary elements to match the SMARTS pattern
    line_suffixes = {} # suffixes of the HETATM lines for each ligand
    atom_num_to_key = {} # atom numbers as keys and keys to ligands as values
    with open(pdb_path, 'rb') as f:
        b_lines = f.readlines()
        for b_line in b_lines:
            if b_line.startswith(b'HETATM'):
                if len(b_line) < 100:
                    return
                if not include_water and b_line[17:20] == b'HOH':
                    continue
                line = b_line.decode('utf-8')
                seg = line[72:76].strip()
                chain = line[21]
                resnum = line[22:26].strip()
                resname = line[17:20].strip()
                element = line[76:78].strip()
                # form the key and construct the ligand block
                key = (biounit, seg, chain, resnum, resname)
                if key not in ligands.keys():
                    ligands[key] = line[:80] + '\n'
                    line_suffixes[key] = [b_line[80:]]
                    has_element[key] = [False] * len(smarts_elements)
                    first_false = 0
                else:
                    ligands[key] += line[:80] + '\n'
                    line_suffixes[key].append(b_line[80:])
                if element in smarts_elements[first_false:]:
                    has_element[key][
                        smarts_elements[first_false:].index(element) + 
                        first_false
                    ] = True
                    if False in has_element[key]:
                        first_false = has_element[key].index(False)
                atom_num = b_line[6:11]
                atom_num_to_key[atom_num] = key
            if b_line.startswith(b'CONECT'):
                atom0 = b_line[6:11]
                if atom0 in atom_num_to_key.keys():
                    line_to_add = 'CONECT' + atom0.decode('utf-8')
                    for i in range(len(b_line[11:]) // 5):
                        atom = b_line[6+5*i:6+5*(i+1)]
                        if atom in atom_num_to_key.keys():
                            line_to_add += atom.decode('utf-8')
                    if len(line_to_add) > 11:
                        ligands[atom_num_to_key[atom0]] += line + '\n'
    for key, value in has_element.items():
        if not all(value):
            ligands.pop(key)
    if len(ligands):
        atom_ids = [b_line[6:11] for b_line in b_lines if b'ATOM' in b_line]
        ent_ids = [b_line[81:85] for b_line in b_lines if b'ATOM' in b_line]
    else:
        return
    
    # Find CGs matching SMARTS pattern
    cg_match_dict = {}
    # if True:
    with suppress_stdout_stderr():
        for key, block in ligands.items():
            # Read ligand block as OBMol object
            mol = ob.OBMol()
            obConversion.ReadString(mol, block)
            mol.PerceiveBondOrders()
            # Match SMARTS pattern to ligand
            if smarts.Match(mol):
                atom_names, contact_ents, all_contact_ents, score = \
                    [], [], set(), None
                for line, suffix in zip(block.split('\n'), 
                                        line_suffixes[key]):
                    if line.startswith('HETATM'):
                        # entity IDs of contact residues
                        atom_names.append(line[12:16].strip())
                        contact_atom_ids = []
                        for i in range(11):
                            contact_atom = suffix[12+9*i:12+9*(i+1)]
                            if contact_atom != b'         ':
                                contact_atom_ids.append(contact_atom[3:8])
                        contact_ents_set = \
                            set([pdb_code + '_' + 
                                 ent_ids[
                                    atom_ids.index(atom_id)
                                 ].decode('utf-8').strip()
                                 for atom_id in contact_atom_ids
                                 if atom_id in atom_ids])
                        contact_ents.append(contact_ents_set)
                        all_contact_ents.update(contact_ents_set)
                        # keep track of the MolProbity score
                        if score is None:
                            try:
                                score = float(suffix[6:10].strip())
                            except:
                                score = float(suffix[1:6].strip())
                # find the lines (i.e. cluster indices) in the 
                # cluster file containing the contacting entities
                cluster_nums = \
                    {ent : find_line_in_file_grep(pdb_cluster_file, ent) 
                     for ent in all_contact_ents if '.' not in ent}
                matching_atoms = smarts.GetUMapList()
                for match in matching_atoms: # 1-indexed atom numbers
                    try:
                        match_names = [atom_names[i - 1] for i in match]
                        match_contact_ent_clusters = tuple(sorted(set(
                            sum([[cluster_nums[ent] 
                                  for ent in contact_ents[i - 1] 
                                  if cluster_nums[ent] >= 0] 
                                 for i in match], [])
                        )))
                        if match_contact_ent_clusters not in \
                                cluster_min_scores.keys() or \
                                cluster_min_scores[
                                    match_contact_ent_clusters
                                ] > score:
                            cluster_min_scores[
                                match_contact_ent_clusters
                            ] = score
                            cg_match_dicts[
                                match_contact_ent_clusters
                            ] = {key : [match_names]}
                        elif cluster_min_scores[
                                    match_contact_ent_clusters
                                ] == score:
                            if key not in cg_match_dicts[
                                match_contact_ent_clusters
                            ].keys():
                                cg_match_dicts[
                                    match_contact_ent_clusters
                                ][key] = [match_names]
                            else:
                                cg_match_dicts[
                                    match_contact_ent_clusters
                                ][key].append(match_names)
                    except:
                        pass