import os
import pickle
from itertools import product

# Constants associated with vdG fingerprint categories
ABPLE_triplets = [''.join(tup) for tup in product('ABPLE', repeat=3)]
relpos = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'same_chain', 'diff_chain']
ABPLE_cols = [str(i) + '_' + ac for i, ac in 
              product(range(1, 11), ABPLE_triplets)]
seqdist_cols = [str(i) + '_' + str(i + 1) + '_' + rp for i, rp in 
                product(range(1, 10), relpos)]
ABPLE_singleton_cols = [col.split('_')[0] + '_' + col.split('_')[1][1] 
                        for col in ABPLE_cols]

# Constants associated with ABPLE assignments
_dir = os.path.dirname(__file__)
path_to_abple_dict = os.path.join(_dir, './files/abple_dict.pkl')
with open(path_to_abple_dict, 'rb') as f:
    abple_dict = pickle.load(f)

# Constants associated with protein sequences and amino acid atoms
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
       'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
       'SER', 'THR', 'TRP', 'TYR', 'VAL']
b_aas = [b'ALA', b'ARG', b'ASN', b'ASP', b'CYS', 
         b'GLN', b'GLU', b'GLY', b'HIS', b'ILE', 
         b'LEU', b'LYS', b'MET', b'PHE', b'PRO', 
         b'SER', b'THR', b'TRP', b'TYR', b'VAL']
non_prot_sel = 'not resname ' + ' and not resname '.join(aas + ['MSE'])
three_to_one = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D',
                'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G',
                'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', 'LYS' : 'K',
                'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S',
                'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V'}
protein_hbond_atoms = {
    'ALA' : ['H', 'O'],
    'ARG' : ['H', 'O', 'HE', 'HH11', 'HH12', 'HH21', 'HH22'], 
    'ASN' : ['H', 'O', 'OD1', 'HD21', 'HD22'],
    'ASP' : ['H', 'O', ('OD1', 'OD2'), 'HD2'],
    'CYS' : ['H', 'O', 'SG', 'HG'],
    'GLN' : ['H', 'O', 'OE1', 'HE21', 'HE22'],
    'GLU' : ['H', 'O', ('OE1', 'OE2'), 'HE2'],
    'GLY' : ['H', 'O'],
    'HIS' : ['H', 'O', 'ND1', 'HD1', 'NE2', 'HE2'],
    'ILE' : ['H', 'O'], 
    'LEU' : ['H', 'O'],
    'LYS' : ['H', 'O', ('HZ1', 'HZ2', 'HZ3')],
    'MET' : ['H', 'O'],
    'PHE' : ['H', 'O'],
    'PRO' : ['H', 'O'],
    'SER' : ['H', 'O', 'OG', 'HG'],
    'THR' : ['H', 'O', 'OG1', 'HG1'],
    'TRP' : ['H', 'O', 'HE1'],
    'TYR' : ['H', 'O', 'OH', 'HH'],
    'VAL' : ['H', 'O']
}
protein_atoms = {
    'ALA' : ['N', 'H', 'CA', 'HA', 'C', 'O', 
             'CB', ('HB1', 'HB2', 'HB3')],
    'ARG' : ['N', 'H', 'CA', 'HA', 'C', 'O', 
             'CB', ('HB2', 'HB3'), 'CG', ('HG2', 'HG3'), 
             'CD', ('HD2', 'HD3'), 'NE', 'HE', 'CZ', 
             'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22'], 
    'ASN' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', 'OD1', 'ND2', 'HD21', 'HD22'],
    'ASP' : ['N', 'H', 'CA', 'HA', 'C', 'O', 
             'CB', ('HB2', 'HB3'), 'CG', ('OD1', 'OD2'), 'HD2'],
    'CYS' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'SG', 'HG'],
    'GLN' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', ('HG2', 'HG3'),
             'CD', 'OE1', 'NE2', 'HE21', 'HE22'],
    'GLU' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', ('HG2', 'HG3'),
             'CD', ('OE1', 'OE2'), 'HE2'],
    'GLY' : ['N', 'H', 'CA', 'HA2', 'HA3', 'C', 'O'],
    'HIS' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', 'ND1', 'HD1', 'CE1', 'HE1',
             'NE2', 'HE2', 'CD2', 'HD2'],
    'ILE' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', 'HB', 'CG2', ('HG21', 'HG22', 'HG23'),
             'CG1', ('HG12', 'HG13'), 'CD1', ('HD11', 'HD12', 'HD13')], 
    'LEU' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', 'HG', 
             'CD1', ('HD11', 'HD12', 'HD13'),
             'CD2', ('HD21', 'HD22', 'HD23')],
    'LYS' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', ('HG2', 'HG3'),
             'CD', ('HD2', 'HD3'), 'CE', ('HE2', 'HE3'),
             'NZ', ('HZ1', 'HZ2', 'HZ3')],
    'MET' : ['N', 'H', 'CA', 'HA', 'C', 'O', 
             'CB', ('HB2', 'HB3'), 'CG', ('HG2', 'HG3'),
             'SD', 'CE', ('HE1', 'HE2', 'HE3')],
    'PHE' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', ('CD1', 'CD2'), ('HD1', 'HD2'),  
             ('CE1', 'CE2'), ('HE1', 'HE2'), 'CZ', 'HZ'],
    'PRO' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', ('HG2', 'HG3'),
             'CD', ('HD2', 'HD3')],
    'SER' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'OG', 'HG'],
    'THR' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', 'HB', 'OG1', 'HG1', 'CG2', ('HG21', 'HG22', 'HG23')],
    'TRP' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', 'CD1', 'HD1', 'NE1', 'HE1',
             'CE2', 'CZ2', 'HZ2', 'CH2', 'HH2', 'CZ3', 'HZ3', 'CE3', 'HE3',
             'CD2', 'HD2'],
    'TYR' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', ('HB2', 'HB3'), 'CG', ('CD1', 'CD2'), ('HD1', 'HD2'), 
             ('CE1', 'HE1'), ('CE2', 'HE2'), 'CZ', 'OH', 'HH'],
    'VAL' : ['N', 'H', 'CA', 'HA', 'C', 'O',
             'CB', 'HB', 'CG1', ('HG11', 'HG12', 'HG13'),
             'CG2', ('HG21', 'HG22', 'HG23')]
}
cg_resnames = {'ccn' : ['LYS'], 
               'gn' : ['ARG'], 
               'coo' : ['ASP', 'GLU'], 
               'cco' : ['SER', 'THR']}
cg_atoms = {'ccn' : {'LYS' : ['CD', 'CE', 'NZ', 'HZ1', 'HZ2', 'HZ3']},
            'gn' : {'ARG' : ['NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12', 
                             'NH2', 'HH21', 'HH22']},
            'coo' : {'ASP' : ['CB', 'CG', 'OD1', 'OD2'],
                     'GLU' : ['CG', 'CD', 'OE1', 'OE2']},
            'cco' : {'SER' : ['CB', 'OG', 'HG'], 
                     'THR' : ['CB', 'OG1', 'HG1']}}
cg_hbond_atoms = {'ccn' : {'LYS' : ['HZ1', 'HZ2', 'HZ3']},
                  'gn' : {'ARG' : ['HE', 'HH11', 'HH12', 'HH21', 'HH22']},
                  'coo' : {'ASP' : ['OD1', 'OD2'],
                           'GLU' : ['OE1', 'OE2']},
                  'cco' : {'SER' : ['OG', 'HG'],
                           'THR' : ['OG1', 'HG1']}}

# constants associated with the periodic table
elements = {
    1: 'H', 2: 'He', 
    3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 
    13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 
    19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 
    37: 'Rb', 38: 'Sr', 
    39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 
    49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 
    55: 'Cs', 56: 'Ba', 
    57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 
    64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 
    76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 
    87: 'Fr', 88: 'Ra', 
    89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 
    96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 
    103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 
    108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn', 
    113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}

# constants associated with ProDy hexadecimal atom IDs
prody_hex_to_decimal = {}
counter = 100000
for i in range(1000000):
    hex_i = hex(i)[2:]
    if i < 100000:
        prody_hex_to_decimal[str(i)] = str(i)
    elif 'a' in hex_i or 'b' in hex_i or 'c' in hex_i or \
            'd' in hex_i or 'e' in hex_i or 'f' in hex_i:
        prody_hex_to_decimal[hex_i] = str(counter)
        counter += 1