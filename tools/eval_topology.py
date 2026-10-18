#!/usr/bin/env python3
"""eval_topology.py - Add contact map overlap and topological metrics.
Usage:
  # Single GPU evaluation (adds contact overlap to lDDT/TM/RMSD)
  python3 eval_topology.py <pred_root> <ref_dir>

  # Cross-GPU comparison (compare two prediction directories)
  python3 eval_topology.py <pred_root_A> <pred_root_B> --compare

Output: <pred_root>/eval_topology_results.json
"""
import sys, json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=RuntimeWarning)

T2O = dict(ALA="A",ARG="R",ASN="N",ASP="D",CYS="C",GLN="Q",GLU="E",
           GLY="G",HIS="H",ILE="I",LEU="L",LYS="K",MET="M",PHE="F",
           PRO="P",SER="S",THR="T",TRP="W",TYR="Y",VAL="V")

def parse_ca(path):
    txt = Path(path).read_text(errors="replace")
    in_atom, col, chains = False, {}, {}
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            if not in_atom: in_atom, col = True, {}
            k = s.split(".",1)[1]
            col[k] = len(col); continue
        if s == "loop_": in_atom, col = False, {}; continue
        if in_atom and col and not s.startswith("#") and not s.startswith("_"):
            t = line.split()
            if len(t) < len(col): continue
            if t[col.get("label_atom_id",-1)] != "CA": continue
            ch = t[col.get("auth_asym_id", col.get("label_asym_id",-1))]
            rn = t[col.get("label_comp_id", col.get("auth_comp_id",-1))]
            try:
                xyz = np.array([float(t[col["Cartn_x"]]),float(t[col["Cartn_y"]]),float(t[col["Cartn_z"]])])
            except (ValueError,KeyError): continue
            chains.setdefault(ch,[]).append((T2O.get(rn,"X"), xyz))
    return chains

def chain_to_array(chains):
    """Convert chain dict {ch: [(aa, xyz), ...]} to flat arrays."""
    all_xyz, all_ch = [], []
    for ch in sorted(chains):
        for _, xyz in chains[ch]:
            all_xyz.append(xyz)
            all_ch.append(ch)
    return np.array(all_xyz), all_ch

def contact_map(xyz, threshold=8.0):
    """Binary contact map: residues within <threshold A (CAs)."""
    n = len(xyz)
    if n < 2: return np.zeros((n,n), dtype=bool)
    dists = np.linalg.norm(xyz[:,None,:] - xyz[None,:,:], axis=2)
    cm = (dists < threshold) & (dists > 0.01)
    return cm

def contact_overlap(cm1, cm2):
    """Compute contact map precision, recall, F1."""
    upper = np.triu_indices_from(cm1, k=1)
    c1 = cm1[upper]
    c2 = cm2[upper]
    tp = np.sum(c1 & c2)
    fp = np.sum(c1 & ~c2)
    fn = np.sum(~c1 & c2)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)

def kabsch(p, q):
    pc, qc = p-p.mean(0), q-q.mean(0)
    U,_,Vt = np.linalg.svd(pc.T@qc)
    d = np.sign(np.linalg.det(Vt.T@U.T))
    R = Vt.T@np.diag([1,1,d])@U.T
    return pc@R.T+q.mean(0)

def lddt(pred, ref):
    L = len(pred)
    if L < 2: return 0.0
    rd = np.linalg.norm(ref[:,None,:]-ref[None,:,:],axis=2)
    pd = np.linalg.norm(pred[:,None,:]-pred[None,:,:],axis=2)
    mask = (rd<15.0)&(rd>0)
    sc = np.zeros(L)
    for i in range(L):
        js = np.where(mask[i])[0]
        if len(js)==0: sc[i]=float("nan"); continue
        diffs = np.abs(pd[i,js]-rd[i,js])
        sc[i] = np.mean([np.mean(diffs<t) for t in (0.5,1.0,2.0,4.0)])
    v = ~np.isnan(sc)
    return float(np.mean(sc[v])) if v.any() else 0.0

def tm_score(pred, ref):
    L = len(pred)
    if L==0: return 0.0
    pa = kabsch(pred, ref)
    d0 = max(1.24*(max(L,19)-15)**(1/3)-1.8, 0.5)
    diff = np.linalg.norm(pa-ref, axis=1)
    return float((1/(1+(diff/d0)**2)).sum()/L)

def rmsd(pred, ref):
    pa = kabsch(pred, ref)
    return float(np.sqrt(((pa-ref)**2).sum()/len(pred)))

def pdbid_of(stem):
    s = stem.lower()
    for i in range(len(s)-3):
        c = s[i:i+4]
        if c[0].isdigit() and c[1:3].isalpha() and (c[3].isdigit() or c[3].isalpha()): return c
    return s[:4]

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    compare_mode = "--compare" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--compare"]

    if compare_mode:
        if len(argv) < 2:
            print("Compare mode: need two prediction directories"); sys.exit(1)
        dir_a, dir_b = Path(argv[0]).resolve(), Path(argv[1]).resolve()
        _compare_two(dir_a, dir_b)
    else:
        pred_root = Path(argv[0]).resolve()
        ref_dir = Path(argv[1]).resolve() if len(argv) > 1 else None
        _eval_single(pred_root, ref_dir)

def _load_all_results(pred_root):
    """Load CIFs from pred_root into (pid -> {sample -> chain_data})."""
    results = {}
    cifs = sorted(pred_root.rglob("*.cif"))
    for cf in cifs:
        if "wounresol" in cf.name: continue
        pid = pdbid_of(cf.stem)
        if pid not in results:
            results[pid] = {}
        results[pid][cf.stem] = parse_ca(cf)
    return results

def _eval_single(pred_root, ref_dir):
    if ref_dir is None:
        print("Need ref_dir for single eval"); sys.exit(1)
    ref_dir = Path(ref_dir)
    refs = {}
    for f in sorted(ref_dir.iterdir()):
        if f.suffix.lower() not in (".cif",".pdb"): continue
        pid = f.stem.lower()[:4]
        refs[pid] = parse_ca(f)
    preds = _load_all_results(pred_root)
    out = {}
    print(f"{'pid':<8} {'sample':<40} {'N':>5} {'lDDT':>8} {'TM':>8} {'RMSD':>8} {'CM8_Prec':>9} {'CM8_Rec':>9} {'CM8_F1':>9} {'CM12_F1':>9}")
    print("-"*120)
    for pid in sorted(preds):
        if pid not in refs: continue
        rc = refs[pid]
        r_xyz, r_ch = chain_to_array(rc)
        r_cm8 = contact_map(r_xyz, 8.0)
        r_cm12 = contact_map(r_xyz, 12.0)
        for sname, pc in sorted(preds[pid].items()):
            p_xyz, p_ch = chain_to_array(pc)
            if len(p_xyz) != len(r_xyz):
                # Different atom counts -> use only overlapping atoms
                min_n = min(len(p_xyz), len(r_xyz))
                p_xyz, r_xyz2 = p_xyz[:min_n], r_xyz[:min_n]
            else:
                r_xyz2 = r_xyz
            if min(len(p_xyz), len(r_xyz2)) < 2: continue
            l = lddt(p_xyz, r_xyz2)
            t = tm_score(p_xyz, r_xyz2)
            rm = rmsd(p_xyz, r_xyz2)
            p_cm8 = contact_map(p_xyz, 8.0)
            p_cm12 = contact_map(p_xyz, 12.0)
            # Subset contact maps to matched atoms
            n = min(p_cm8.shape[0], r_cm8.shape[0])
            p8, r8 = p_cm8[:n,:n], r_cm8[:n,:n]
            p12, r12 = p_cm12[:n,:n], r_cm12[:n,:n]
            pr, rc_, f1_8 = contact_overlap(p8, r8)
            _, _, f1_12 = contact_overlap(p12, r12)
            out.setdefault(pid,[]).append({
                "sample": sname, "n": n,
                "lddt": round(l,4), "tm": round(t,4), "rmsd": round(rm,3),
                "cm8_prec": round(pr,4), "cm8_rec": round(rc_,4),
                "cm8_f1": round(f1_8,4), "cm12_f1": round(f1_12,4),
            })
            print(f"{pid:<8} {sname:<40} {n:>5} {l:>8.4f} {t:>8.4f} {rm:>8.3f} {pr:>9.4f} {rc_:>9.4f} {f1_8:>9.4f} {f1_12:>9.4f}")
    # Summary
    print("\n=== Summary (per target) ===")
    for pid, vals in sorted(out.items()):
        a = np.array([v["lddt"] for v in vals])
        f1s = np.array([v["cm8_f1"] for v in vals])
        print(f"  {pid}: lDDT={a.mean():.4f}[best={a.max():.4f}]  CM8_F1={f1s.mean():.4f}  n={len(vals)}")
    (pred_root/"eval_topology_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResults saved: {pred_root/'eval_topology_results.json'}")

def _compare_two(dir_a, dir_b):
    """Compare two prediction directories (e.g. 910B vs 910C)."""
    preds_a = _load_all_results(dir_a)
    preds_b = _load_all_results(dir_b)
    common_pids = set(preds_a) & set(preds_b)
    if not common_pids:
        print("No common targets between the two directories"); return
    print(f"Comparing {len(common_pids)} common targets\n")
    rows = []
    for pid in sorted(common_pids):
        for sname in sorted(set(preds_a[pid]) & set(preds_b[pid])):
            pa = preds_a[pid][sname]
            pb = preds_b[pid][sname]
            xyz_a, _ = chain_to_array(pa)
            xyz_b, _ = chain_to_array(pb)
            n = min(len(xyz_a), len(xyz_b))
            if n < 2: continue
            xa, xb = xyz_a[:n], xyz_b[:n]
            # Direct comparison between two predictions
            rm = rmsd(xa, xb)
            cm_a = contact_map(xa, 8.0)
            cm_b = contact_map(xb, 8.0)
            _, _, f1_8 = contact_overlap(cm_a, cm_b)
            # Per-atom distance
            pa_ = kabsch(xa, xb)
            per_atom_d = np.linalg.norm(pa_ - xb, axis=1).mean()
            rows.append({
                "pid": pid, "sample": sname,
                "n_atom": n,
                "pairwise_rmsd": round(rm, 3),
                "pairwise_cm8_f1": round(f1_8, 4),
                "per_atom_mean_dist": round(float(per_atom_d), 4),
            })
            print(f"{pid:<8} {sname:<40} RMSD={rm:.3f}  CM8_F1={f1_8:.4f}  mean_dist={per_atom_d:.4f}")
    summary = {"targets": len(common_pids), "pairs": len(rows), "results": rows}
    (dir_a/"compare_topology_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved: {dir_a/'compare_topology_results.json'}")

if __name__=="__main__":
    main()
