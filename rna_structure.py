import os
import argparse
import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq

# Import proto_tools as used in the notebook
from proto_tools import ViennaRNAInput, ViennaRNAConfig, run_viennarna

def extract_and_mutate(row, genome, flank):
    variant = row["variant"]
    strand = row["strand"]

    # Expected format: chr17:43045712:T>C
    chrom, pos, alleles = variant.split(":")
    pos = int(pos)
    ref, mut = alleles.split(">")

    # FASTA uses 1, 2, 3... not chr1, chr2...
    chrom = chrom.replace("chr", "")

    # Extract genomic (+ strand)
    left = pos - flank - 1
    right = pos + len(ref) - 1 + flank
    seq = genome[chrom][left:right].seq.upper()

    # Variant position in extracted sequence
    idx = flank

    # Verify reference allele
    genome_ref = seq[idx:idx + len(ref)]
    assert genome_ref == ref, f"{variant}: reference mismatch. Expected {ref}, found {genome_ref}"

    # Apply mutation
    mut_seq = seq[:idx] + mut + seq[idx + len(ref):]

    # Confirm length logic (as per notebook)
    expected_length_change = len(mut) - len(ref)
    assert len(mut_seq) == len(seq) + expected_length_change, (
        f"{variant}: unexpected length change. Expected {expected_length_change}, got {len(mut_seq) - len(seq)}"
    )

    # Convert to transcript orientation
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
        mut_seq = str(Seq(mut_seq).reverse_complement())

    return pd.Series({
        "variant": variant,
        "strand": strand,
        "ref_seq": seq,
        "mut_seq": mut_seq
    })

def main():
    parser = argparse.ArgumentParser(description="Extract, mutate and fold sequences via ViennaRNA.")
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV (Format: variant,strand)")
    parser.add_argument("-f", "--flank", type=int, required=True, help="Flank size (e.g., 50)")
    parser.add_argument("-o", "--output_folder", required=True, help="Folder to save the final CSV")
    parser.add_argument("-g", "--genome", required=True, help="Path to reference genome FASTA file")
    parser.add_argument("-d", "--device", default="cpu", help="Device to run inference on (e.g., 'cpu', 'gpu')")
    
    args = parser.parse_args()

    # Read the input CSV (Assuming no header based on your description)
    # If your file has headers, you can drop `names=...` and use `header=0`
    df = pd.read_csv(args.input, names=["variant", "strand"])

    # Load reference genome
    print(f"Loading reference genome from {args.genome}...")
    genome = Fasta(args.genome)

    print("Extracting and mutating sequences...")
    # Apply mutation function
    sequence_df = df.apply(lambda row: extract_and_mutate(row, genome, args.flank), axis=1)

    ref_seqs = sequence_df["ref_seq"].tolist()
    mut_seqs = sequence_df["mut_seq"].tolist()

    # Initialize ViennaRNA configurations
    config = ViennaRNAConfig(
        temperature=37.0,  
        verbose=1,
        device=args.device  # Running on the specified device
    )

    print("Running ViennaRNA prediction for reference sequences...")
    ref_inputs = ViennaRNAInput(sequences=ref_seqs)
    ref_result = run_viennarna(ref_inputs, config)

    print("Running ViennaRNA prediction for mutated sequences...")
    mut_inputs = ViennaRNAInput(sequences=mut_seqs)
    mut_result = run_viennarna(mut_inputs, config)

    print("Merging dataframes...")
    # Map the object result records directly to a dictionary structure matching your required 6 columns
    ref_records = [{
        "ref_sequence": r.sequence, 
        "ref_structure": r.structure, 
        "ref_mfe": r.mfe
    } for r in ref_result.results]
    
    mut_records = [{
        "mut_sequence": r.sequence, 
        "mut_structure": r.structure, 
        "mut_mfe": r.mfe
    } for r in mut_result.results]

    ref_df = pd.DataFrame(ref_records)
    mut_df = pd.DataFrame(mut_records)

    # Concatenate the resulting outputs column-wise
    merged_df = pd.concat([ref_df, mut_df], axis=1)

    # Export to CSV
    merged_df.to_csv(os.path.join(args.output_folder, 'vienna_results.csv'), index=False)
    print(f"Done! Saved 6 columns to {args.output_folder}")

if __name__ == "__main__":
    main()