#!/usr/bin/env encoding=utf-8
"""
RNA 3D PDB Generator from ViennaRNA CSV Output
Author: Antigravity

This script automates the conversion of RNA sequences and dot-bracket secondary structures
from a results CSV file into individual 3D structural .pdb files.

It uses a spring-embedder (force-directed) relaxation model in 3D to simulate the physical
constraints of the RNA chain:
  1. Sequential connectivity: residues are spaced ~4.0 A apart along the backbone.
  2. Base-pairing constraints: paired nucleotides (from dot-bracket) are pulled together to ~2.8 A.
  3. Collision avoidance: non-bonded residues repel each other to prevent steric overlaps.
  4. Center gravity: a gentle force keeps the molecule centered at the origin.

The output PDB files contain:
  - 'P' and 'C4\'' atoms representing the backbone (which 3Dmol.js renders as a cartoon ribbon).
  - 'N1' (pyrimidines: C, U) or 'N9' (purines: A, G) atoms representing base positions,
    placed facing their pairing partner to be rendered as base-pairing sticks.
"""

import os
import csv
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

# Standard residue definitions
PURINES = {'A', 'G'}
PYRIMIDINES = {'C', 'U', 'T'}

def parse_dot_bracket(structure: str) -> Dict[int, int]:
    """
    Parses secondary structure in dot-bracket notation (supporting brackets and braces
    for pseudoknots) and returns a mapping of base pairs.
    """
    bp_map = {}
    paren_stack = []
    bracket_stack = []
    brace_stack = []
    
    for i, char in enumerate(structure):
        if char == '(':
            paren_stack.append(i)
        elif char == ')':
            if paren_stack:
                j = paren_stack.pop()
                bp_map[i] = j
                bp_map[j] = i
        elif char == '[':
            bracket_stack.append(i)
        elif char == ']':
            if bracket_stack:
                j = bracket_stack.pop()
                bp_map[i] = j
                bp_map[j] = i
        elif char == '{':
            brace_stack.append(i)
        elif char == '}':
            if brace_stack:
                j = brace_stack.pop()
                bp_map[i] = j
                bp_map[j] = i
                
    return bp_map

def generate_folded_coordinates(sequence: str, bp_map: Dict[int, int], iterations: int = 150) -> List[Tuple[float, float, float]]:
    """
    Generates 3D coordinates for the RNA backbone residues using a force-directed model.
    """
    n = len(sequence)
    if n == 0:
        return []
        
    # 1. Initialize coordinates on a helix-like cylinder to prevent initial overlap
    coords = []
    for i in range(n):
        angle = 2.0 * math.pi * i / max(10, n * 0.3)
        r = 8.0 + n * 0.05
        z = (i - n / 2) * 1.5
        coords.append([
            r * math.cos(angle) + random.uniform(-0.2, 0.2),
            r * math.sin(angle) + random.uniform(-0.2, 0.2),
            z + random.uniform(-0.2, 0.2)
        ])
        
    # 2. Force relaxation loop
    dt = 0.15 # Time step
    for _ in range(iterations):
        forces = [[0.0, 0.0, 0.0] for _ in range(n)]
        
        # Force A: Sequential backbone bonding (target: 4.0 A)
        for i in range(n - 1):
            p1, p2 = coords[i], coords[i+1]
            dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 0.01
            diff = dist - 4.0
            
            # Spring force
            fx = (dx / dist) * diff * 0.8
            fy = (dy / dist) * diff * 0.8
            fz = (dz / dist) * diff * 0.8
            
            forces[i][0] += fx
            forces[i][1] += fy
            forces[i][2] += fz
            forces[i+1][0] -= fx
            forces[i+1][1] -= fy
            forces[i+1][2] -= fz
            
        # Force B: Base-pairing contacts (target: 2.8 A)
        for i, j in bp_map.items():
            if i < j:
                p1, p2 = coords[i], coords[j]
                dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 0.01
                diff = dist - 2.8
                
                # Stronger spring force
                fx = (dx / dist) * diff * 1.8
                fy = (dy / dist) * diff * 1.8
                fz = (dz / dist) * diff * 1.8
                
                forces[i][0] += fx
                forces[i][1] += fy
                forces[i][2] += fz
                forces[j][0] -= fx
                forces[j][1] -= fy
                forces[j][2] -= fz
                
        # Force C: Steric repulsion (prevent residue collision)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if already bonded sequentially or base-paired
                if j == i + 1 or (i in bp_map and bp_map[i] == j):
                    continue
                p1, p2 = coords[i], coords[j]
                dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 0.01
                
                # If closer than 6.5 A, repel
                if dist < 6.5:
                    repel = 6.5 - dist
                    fx = (dx / dist) * repel * 0.4
                    fy = (dy / dist) * repel * 0.4
                    fz = (dz / dist) * repel * 0.4
                    
                    forces[i][0] -= fx
                    forces[i][1] -= fy
                    forces[i][2] -= fz
                    forces[j][0] += fx
                    forces[j][1] += fy
                    forces[j][2] += fz
                    
        # Force D: Center gravity (keeps the structure near origin)
        for i in range(n):
            forces[i][0] -= coords[i][0] * 0.015
            forces[i][1] -= coords[i][1] * 0.015
            forces[i][2] -= coords[i][2] * 0.015
            
        # 3. Update coordinates based on forces
        for i in range(n):
            # Cap maximum force displacement to avoid numerical explosion
            fx, fy, fz = forces[i]
            f_norm = math.sqrt(fx*fx + fy*fy + fz*fz)
            if f_norm > 4.0:
                fx = (fx / f_norm) * 4.0
                fy = (fy / f_norm) * 4.0
                fz = (fz / f_norm) * 4.0
                
            coords[i][0] += fx * dt
            coords[i][1] += fy * dt
            coords[i][2] += fz * dt
            
    return [(c[0], c[1], c[2]) for c in coords]

def write_pdb_structure(
    filepath: str, 
    sequence: str, 
    bp_map: Dict[int, int],
    coords: List[Tuple[float, float, float]]
):
    """
    Writes spatial coordinates to a standard PDB format file.
    Creates:
      - P: Phosphate atom on backbone
      - C4': Backbone sugar carbon
      - N1/N9: Base nitrogen pointing towards its base-pair partner
    """
    n = len(sequence)
    atom_idx = 1
    
    with open(filepath, 'w') as f:
        f.write(f"HEADER    RNA 3D PREDICTION MODEL - LENGTH {n} NT\n")
        f.write(f"TITLE     GENERATED BY ANTIGRAVITY FORCE-FIELD RELAXATION MODEL\n")
        
        # Center of mass calculation for pointing unpaired bases
        cx = sum(c[0] for c in coords) / n
        cy = sum(c[1] for c in coords) / n
        cz = sum(c[2] for c in coords) / n
        
        for i in range(n):
            res_name = sequence[i].upper()
            res_seq = i + 1
            x, y, z = coords[i]
            
            # 1. Write Backbone Phosphate (P)
            # PDB ATOM format: ATOM, serial, name, altLoc, resName, chainID, resSeq, iCode, x, y, z, occupancy, tempFactor, element
            f.write(f"ATOM  {atom_idx:>5}  {'P':<3} {res_name:>3} A{res_seq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {'P':>2}\n")
            atom_idx += 1
            
            # 2. Write Sugar Carbon (C4') slightly offset from Phosphate
            f.write(f"ATOM  {atom_idx:>5}  {'C4\'':<3} {res_name:>3} A{res_seq:>4}    {x+0.8:8.3f}{y+0.8:8.3f}{z+0.8:8.3f}  1.00 20.00           {'C':>2}\n")
            atom_idx += 1
            
            # 3. Write Base Nitrogen (N1 or N9) pointing towards pairing partner or center
            base_atom = 'N9' if res_name in PURINES else 'N1'
            
            if i in bp_map:
                # Point base towards its partner j
                partner_idx = bp_map[i]
                px, py, pz = coords[partner_idx]
                dx, dy, dz = px - x, py - y, pz - z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 0.01
                
                # Base is placed 1.2 A along the vector towards partner
                bx = x + 1.2 * (dx / dist)
                by = y + 1.2 * (dy / dist)
                bz = z + 1.2 * (dz / dist)
            else:
                # Point base inwards towards the center of mass
                dx, dy, dz = cx - x, cy - y, cz - z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz) or 0.01
                bx = x + 1.2 * (dx / dist)
                by = y + 1.2 * (dy / dist)
                bz = z + 1.2 * (dz / dist)
                
            f.write(f"ATOM  {atom_idx:>5}  {base_atom:<3} {res_name:>3} A{res_seq:>4}    {bx:8.3f}{by:8.3f}{bz:8.3f}  1.00 20.00           {base_atom[:1]:>2}\n")
            atom_idx += 1
            
        f.write("END\n")

def main():
    parser = argparse.ArgumentParser(description="Fold RNA sequences and secondary structures into 3D PDB coordinates.")
    parser.add_argument(
        "--csv", "-i", 
        default="viennarna_results.csv", 
        help="Path to the input results.csv file (default: viennarna_results.csv)"
    )
    parser.add_argument(
        "--output", "-o", 
        default="rna-dashboard/public/static/pdbs", 
        help="Target folder for PDB files (default: rna-dashboard/public/static/pdbs)"
    )
    parser.add_argument(
        "--limit", "-l", 
        default="all", 
        help="Number of structures to generate (default: all, or enter an integer e.g., 50)"
    )
    parser.add_argument(
        "--iterations", "-n", 
        type=int, 
        default=150, 
        help="Number of force-relaxation iterations (default: 150)"
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv
    output_dir = args.output
    
    if not os.path.exists(csv_path):
        print(f"Error: Input file '{csv_path}' not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Read rows
    rows = []
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Verify columns
            required_cols = {'sequence', 'structure', 'mfe'}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                print(f"Error: CSV headers {reader.fieldnames} must contain 'sequence', 'structure', and 'mfe'.")
                return
                
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    total_available = len(rows)
    print(f"Found {total_available} folding predictions in '{csv_path}'.")
    
    # Parse limit
    if args.limit.lower() == 'all':
        limit = total_available
    else:
        try:
            limit = min(int(args.limit), total_available)
        except ValueError:
            print(f"Invalid limit value '{args.limit}'. Defaulting to all.")
            limit = total_available
            
    print(f"Generating 3D models for first {limit} entries in directory: {output_dir}")
    
    # Generate structures
    success_count = 0
    for idx in range(limit):
        row = rows[idx]
        seq = row['sequence'].strip()
        struct = row['structure'].strip()
        mfe = row['mfe'].strip()
        
        filename = f"structure_{idx}.pdb"
        filepath = os.path.join(output_dir, filename)
        
        try:
            bp_map = parse_dot_bracket(struct)
            coords = generate_folded_coordinates(seq, bp_map, iterations=args.iterations)
            write_pdb_structure(filepath, seq, bp_map, coords)
            success_count += 1
            
            # Simple progress log
            if (idx + 1) % 50 == 0 or (idx + 1) == limit:
                pct = ((idx + 1) / limit) * 100
                print(f"Progress: {idx + 1}/{limit} files written ({pct:.1f}%)")
                
        except Exception as e:
            print(f"Error processing row {idx} ({seq[:10]}...): {e}")
            
    print(f"\nCompleted! Successfully folded and saved {success_count} structures to '{output_dir}'.")

if __name__ == "__main__":
    main()
