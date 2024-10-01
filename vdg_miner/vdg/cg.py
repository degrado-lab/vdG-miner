
import os
import sys
import gzip
import subprocess

from openbabel import openbabel as ob

from vdg_miner.constants import b_aas

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
    line : str
        Line in the file containing the target string.
    """
    return subprocess.run(['grep', '-n', target_string, file_path], 
                          stdout=subprocess.PIPE, text=True).stdout.strip()

def find_cg_matches(smarts_pattern, pdb_path, pdb_cluster_file, 
                    include_water=False):
    """
    Find CGs matching a SMARTS pattern in PDB files.
    
    Parameters
    ----------
    smarts_pattern : str
        SMARTS pattern.
    pdb_path : str
        Path to directory containing PDB files organized in subdirectories 
        by the middle two characters of the PDB ID.
    pdb_cluster_file : str
        Path to file containing the RCSB-curated sequence clusters in the PDB.
    include_water : bool, optional
        Whether to include water molecules in the search. Default is False.
    
    Returns
    -------
    cg_match_dict : dict, optional
        Dictionary of matching CGs in ligands, with tuples of 
        (struct_name, seg, chain, resnum, resname) for the ligand as keys 
        and, as values, lists that contain the list of atom names, the 
        MolProbity score, and the set of cluster numbers of the contacting 
        entities for each match to the CG. Used for non-protein CGs.
    """
    pdb_code = pdb_path[:4].upper()
    biounit = pdb_path.split('/')[-1][:-4]
    # Read PDB file and extract ligands as blocks
    ligands = {}
    atom_nums = {}
    with open(pdb_path, 'rb') as f:
        b_lines = f.readlines()
        atom_ids = [b_line[6:11].strip() for b_line in b_lines]
        for b_line in b_lines:
            if b_line.startswith(b'HETATM'):
                line = b_line.decode('utf-8')
                resname = line[17:20].strip()
                if not include_water and resname == 'HOH':
                    continue
                seg = line[72:76].strip()
                chain = line[21]
                resnum = line[22:26].strip()
                atom_num = line[6:11].strip()
                # entity IDs of contacting residues and MolProbity score
                contact_atom_ids = []
                for i in range(12):
                    contact_atom = b_line[100+9*i:100+9*(i+1)].strip()
                    if contact_atom != b'         ':
                        contact_atom_ids.append(contact_atom[4:])
                contact_ents = ' '.join(
                    set([pdb_code + '_' + 
                         atom_ids.index(atom_id).decode('utf-8') 
                         for atom_id in contact_atom_ids])
                )
                # form the key and construct the ligand block
                key = (biounit, seg, chain, resnum, resname)
                if key not in ligands.keys():
                    ligands[key] = line[:100] + contact_ents
                else:
                    ligands[key] += line[:100] + contact_ents
                atom_nums[atom_num] = key
            if b_line.startswith(b'CONECT'):
                line = b_line.decode('utf-8')
                atom0 = line.split()[1]
                if atom0 in atom_nums.keys():
                    line_to_add = 'CONECT' + atom0.rjust(5)
                    for atom in line.split()[2:]:
                        if atom in atom_nums.keys():
                            line_to_add += atom.rjust(5)
                    if len(line_to_add) > 11:
                        ligands[atom_nums[atom0]] += line

    # Initialize Open Babel conversion object
    obConversion = ob.OBConversion()
    obConversion.SetInFormat('pdb')

    # Initialize Open Babel SMARTS matcher
    smarts = ob.OBSmartsPattern()
    smarts.Init(smarts_pattern)
    
    # Find CGs matching SMARTS pattern
    cg_match_dict = {}
    with suppress_stdout_stderr():
        for key, block in ligands.items():
            # Read ligand block as OBMol object
            mol = ob.OBMol()
            obConversion.ReadString(mol, block)
            mol.PerceiveBondOrders()
            # Match SMARTS pattern to ligand
            if smarts.Match(mol):
                atom_names, contact_ent_clusters, score = [], [], None
                for line in block.split('\n'):
                    if line.startswith('HETATM'):
                        # find the lines (i.e. cluster indices) in the 
                        # cluster file containing the contacting entities
                        cluster_nums = \
                            [find_line_in_file_grep(pdb_cluster_file, ent) 
                             for ent in line[100:].split()]
                        atom_names.append(line[12:16].strip())
                        contact_ent_clusters.append(cluster_nums)
                        if score is None:
                            score = float(line[86:91].strip())
                matching_atoms = smarts.GetUMapList()
                for match in matching_atoms: # 1-indexed atom numbers
                    try:
                        match_names = [atom_names[i - 1] for i in match]
                        match_contact_ent_clusters = set(
                            sum([contact_ent_clusters[i - 1] 
                                 for i in match], [])
                        )
                        match_info = [match_names, score, 
                                      match_contact_ent_clusters]
                        if key not in cg_match_dict.keys():
                            cg_match_dict[key] = [match_info]
                        else:
                            cg_match_dict[key].append(match_info)
                    except:
                        pass

    return cg_match_dict


def filter_cg_matches(cg_match_dict):
    """Filter the CG matches to remove matches from redundant structures.

    Parameters
    ----------
    cg_match_dict : dict
        Dictionary of matching CGs in ligands, with tuples of 
        (struct_name, seg, chain, resnum, resname) for the ligand as keys 
        and, as values, lists that contain the list of atom names, the 
        MolProbity score, and the set of cluster numbers of the contacting 
        entities for each match to the CG. Used for non-protein CGs.
    
    Returns
    -------
    filtered_cg_match_dict : dict
        Dictionary of matching CGs in ligands, with tuples of 
        (struct_name, seg, chain, resnum, resname) for the ligand as keys 
        and, as values, lists that contain the list of atom names for each 
        non-redundant match to the CG. Used for non-protein CGs.
    """
    return