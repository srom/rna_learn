import os
import re

import pandas as pd
import numpy as np


def load_rna_structure_dataset(metadata, sequence_folder_path):
    sequences = []
    for tpl in metadata.itertuples():
        rna_type, prot_type = getattr(tpl, 'category').split(' ')
        filename = tpl.sp.replace(' ', '_') + '.structure.txt'

        path = os.path.join(
            sequence_folder_path, 
            rna_type, 
            prot_type,
            filename,
        )
        with open(path) as f:
            content = f.read()

            # Remove free energy information at the end
            content = re.sub(
               r'\s+\([-0-9\.]+\)\s*$', 
               '', 
               content
            )
            content = content.strip()

            sequences.append(content)

    return sequences
