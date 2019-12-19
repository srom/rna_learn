RNA structure vs Optimal Growth Temperature (OGT)
--------------------------------------------------

# Review

Structural signatures of thermal adaptation of bacterial ribosomal RNA, transfer RNA, and messenger RNA

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5598986/

Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/9DDRNU


# Preprocessing

Extract RNA 2D structure using the RNAfold package of [ViennaRNA](https://www.tbi.univie.ac.at/RNA/).

```bash
./preprocessing/generate_mfe_structure.sh "data/seq/mrna/**/*.fasta"
```

The script iterates through files matching the pattern provided and creates one text file in the same location with extension `.structure.txt`.

Output file example:

```
..................(((((((((((((((.....(((((..((((((.....((((.((((((((......(((..(((.(((((..((((((..(((((..(((((((.......(((((((((......))))).).)))...((((((.......))))))......(((((.(((.........)))))))).(((((((..(((((((...((((((.((........)))))))).)))))))......))))))).))))))).)))))...))))))..)))))..)))..)))))))))))...))))..(((((((.........))))))).(((...((((((((((....((((...)))).((((((((.(((.....))))))))))).((((((.......((.(((((.((((....)))).)))))..)).....))))))(((...........)))..))))))))))...))).(((.(((.....))))))..(((...((((((((((.......))))))))))...)))...).)))))..)))))....)))))))((((.(((((.(((((((.((((...)))).)))))))..))))))))).))))))))....... (-176.60)
```
