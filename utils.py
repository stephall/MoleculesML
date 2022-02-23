#utils.py

import numpy as np
from rdkit import Chem
from IPython.display import Image

def draw_molecule(smiles_string, graphic_path='images/to_draw.png'):
    """
    Draw/display the molecule represented by its input smiles string.
    
    Args:
        smiles_string (str): Molecule represented by its smiles string.
        graphic_path: Path where the graphic file of the drawn molecule will 
            be stored. (Default 'images/to_draw.png')
    
    Outputs:
        None
    """
    # Check that the input is a string
    if not isinstance(smiles_string, str):
        err_msg = f"The first argument to 'draw_molecule' must be the smiles string of the " \
                  f"molecule, got  input of type {type(smiles_string)} instead."
        raise TypeError(err_msg)

    # Make a molecule object via RDKit from the smiles string
    molecule_chem_obj = Chem.MolFromSmiles(smiles_string)

    # Draw the molecule
    Chem.Draw.MolToFile(molecule_chem_obj, graphic_path)

    # Show the image of the molecule
    return Image(filename=graphic_path)


