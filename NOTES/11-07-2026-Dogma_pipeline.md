## Dogma pipeline 11-07-2026

In DOGMA there are three modalities:

1. DNA
2. RNA
3. Protein

Currently they are in one notebook files and not connected and no UI interface. How should I structure the files so they connect and also for a user interface.

IFor a quick test-case of the user interface. You puy in a location e.g. different box as shown below:

- variant_chromosome = 'chr22' # @param { type:"string" }
- variant_position = 36201698 # @param { type:"integer" }
- variant_reference_bases = 'A' # @param { type:"string" }
- variant_alternate_bases = 'C' # @param { type:"string" }

Given this postions it goes through the Alphagenome code and tries to run it. In addition, you can choose the ontology_curie you want and also the type of TRACKS you want to pick for you can choose multiple or one like a checkbox e.g. either ATAC, RNA_seq etc.

The output is stored and displayed on the UI in a table format.

For the flow of sequence to ViennaRNA and ESM from the original coordinate and to be used as input sequence for these softwares. Please analyse carefully and check if this is correc and if my logic is correct.

1. For ViennaRNA use the coordinate similar to alphagenome and have a sliding window in the UI to see how many bp to extract on either end of the given position. Oce the sequence is extracted use that sequence to feed into the ViennaRNA code.

2. For ESM you want to index the gene_name column in this case it will be like APOL4 for each fo the isoforms you put that into ESM score with reference and mutation and you print out the sequence ans the isoform as well into the UI.

**GIVE ME THE FULL CODE PLEASE AND HOW TO STRUCTURE THE FILES**

### What else should the UI contain:

1. Place to put the Alphagenome API key
2. Sliding window to extract how many bp to extract on either end of the DNA sequence to be fed into ViennaRNA
3. Place at the bottom that prints out the table for alphagenome, scores from ViennaRNA and ESM scores.
