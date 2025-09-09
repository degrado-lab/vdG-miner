# vdG-miner
Extract van der Graphs from a mirror of the PDB.

## Description

This code generates van der Graphs (vdGs), a structural unit that consists of 
a chemical group (CG), either from within a protein side chain or a ligand, 
and the constellation of residues that interact with the CG, either directly 
or via water-mediated hydrogen bonds. vdGs are clusters of such constellations 
whose sizes roughly correlate with the favorability of the associated 
interactions, and are thus useful in the design of ligand-binding proteins. 
vdGs generated from ligands can be defined using SMARTS patterns, allowing 
vdGs to represent extremely diverse forms of intermolecular interactions. The 
associated [vdG-designer](https://github.com/degrado-lab/vdG-designer) module 
provides tools to generate ligand-binding pockets from vdGs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

vdG-designer requires packages to be installed via the Conda package manager.

First, clone this repository:
```bash
git clone https://github.com/degrado-lab/vdG-designer.git
```

Inside the repository, create a new conda environment using the provided file 
(note that this file is redundant with the environment YML file for 
[vdG-designer](https://github.com/degrado-lab/vdG-designer)):
```bash
conda env create -f environment.yml -y
```

Then, activate the environment:
```bash
conda activate env_prody

### using the code
This code is intended to generate the necessary pickled objects for the generation of van der Graph 
(vdG) databases to aid in the design of ligand-binding proteins.  A mirror of the PDB and validation 
information for each structure is a necessary prerequisite.  These can be downloaded from the 
PDB FTP server as follows:

```bash
> rsync -rlpt -v -z --delete --port=33444
  rsync.rcsb.org::ftp_data/structures/divided/pdb/ $LOCAL_PDB_MIRROR_PATH
  
> rsync -rlpt -v -z --delete --include="*/" --include="*.xml.gz" --exclude="*"
  --port=33444 rsync.rcsb.org::ftp/validation_reports/ $LOCAL_VALIDATION_PATH
```


