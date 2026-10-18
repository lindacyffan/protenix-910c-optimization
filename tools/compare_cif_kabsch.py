#!/usr/bin/env python3
"""Compare two mmCIF models by CA and all-atom coordinates."""

from __future__ import annotations

import argparse
import json
import math

import gemmi
import numpy as np


def atom_table(path: str):
    structure = gemmi.read_structure(path)
    structure.setup_entities()
    model = structure[0]
    ca = {}
    atoms = {}
    for chain in model:
        for residue in chain:
            seq = residue.seqid
            for atom in residue:
                key = (chain.name, seq.num, seq.icode, atom.name)
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                atoms[key] = xyz
                if atom.name == "CA":
                    ca.setdefault((chain.name, seq.num, seq.icode), xyz)
    return ca, atoms


def paired_ca(a: dict, b: dict):
    keys = sorted(a.keys() & b.keys())
    if not keys:
        raise ValueError("No paired CA atoms")
    return np.stack([a[key] for key in keys]), np.stack([b[key] for key in keys])


def kabsch(source: np.ndarray, target: np.ndarray):
    src_center = source.mean(axis=0)
    dst_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - src_center).T @ (target - dst_center))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    translation = dst_center - src_center @ rotation
    return rotation, translation


def rmsd(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))


def tm_score(source: np.ndarray, target: np.ndarray) -> float:
    distance = np.linalg.norm(source - target, axis=1)
    length = len(source)
    d0 = max(0.5, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)
    return float(np.sum(1.0 / (1.0 + (distance / d0) ** 2)) / length)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_cif")
    parser.add_argument("target_cif")
    parser.add_argument("--json")
    args = parser.parse_args()

    source_ca_all, source_atoms = atom_table(args.source_cif)
    target_ca_all, target_atoms = atom_table(args.target_cif)
    source_ca, target_ca = paired_ca(source_ca_all, target_ca_all)
    rotation, translation = kabsch(source_ca, target_ca)
    source_ca_aligned = source_ca @ rotation + translation

    atom_keys = sorted(source_atoms.keys() & target_atoms.keys())
    source_all = np.stack([source_atoms[key] for key in atom_keys])
    target_all = np.stack([target_atoms[key] for key in atom_keys])
    source_all_aligned = source_all @ rotation + translation

    result = {
        "source_cif": args.source_cif,
        "target_cif": args.target_cif,
        "paired_ca": int(len(source_ca)),
        "paired_all_atom": int(len(atom_keys)),
        "ca_rmsd_before_alignment_a": rmsd(source_ca, target_ca),
        "ca_rmsd_after_alignment_a": rmsd(source_ca_aligned, target_ca),
        "all_atom_rmsd_under_ca_transform_a": rmsd(
            source_all_aligned, target_all
        ),
        "ca_tm_score": tm_score(source_ca_aligned, target_ca),
        "all_atom_under_0p5_a_percent": float(
            100.0
            * np.mean(
                np.linalg.norm(source_all_aligned - target_all, axis=1) < 0.5
            )
        ),
        "all_atom_under_1a_percent": float(
            100.0
            * np.mean(np.linalg.norm(source_all_aligned - target_all, axis=1) < 1.0)
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
