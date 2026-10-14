"""Digitize the MiniBooNE dipole-portal figure (arXiv:2206.07100 Fig. 'GlobalWithoutMB')."""
import numpy as np
from extract_paths import parse, content_stream

PDF = str(__import__("pathlib").Path(__file__).with_name("GlobalWithoutMB_ConfidenceRegions.pdf"))
X0, XSTEP = 97.1465, 256.9169      # device x of 10 MeV, pt per decade
Y0, YSTEP = 178.7748, 178.2894     # device y of 1e-7, pt per decade
BOX = (97.1465, 688.32, 71.4339, 535.3536)
LEGEND = (380.0, 688.0, 71.0, 165.0)   # device bbox of the legend

def to_data(pts):
    a = np.asarray(pts, float)
    m = 10.0 ** (1.0 + (a[:, 0] - X0) / XSTEP)      # MeV
    d = 10.0 ** (-7.0 + (a[:, 1] - Y0) / YSTEP)     # GeV^-1
    return np.column_stack([m, d])

def in_legend(a):
    return (a[:, 0].min() > LEGEND[0] and a[:, 0].max() < LEGEND[1]
            and a[:, 1].min() > LEGEND[2] and a[:, 1].max() < LEGEND[3])

def collect(color, dash, min_pts=8):
    out = []
    for p in parse(content_stream(PDF)):
        if p["kind"] != "S" or p["dash"] != dash: continue
        if max(abs(np.array(p["color"]) - np.array(color))) > 0.02: continue
        for sp in p["subpaths"]:
            a = np.asarray(sp, float)
            if len(a) < min_pts or in_legend(a): continue
            out.append(to_data(a))
    return out

PINK  = (0.768319877, 0.1038831219, 0.4868896578)
GREEN = (0.2990388312, 0.5690119185, 0.1287966167)
GRAY  = (0.5019607843,)*3

CURVES = {
    "enuqe_3sigma": collect(PINK, "solid"),
    "enuqe_95":     collect(PINK, "dashed"),
    "costheta_3sigma": collect(GREEN, "solid"),
    "costheta_95":     collect(GREEN, "dashed"),
    "constraints":     collect(GRAY, "solid", min_pts=4),
}

if __name__ == "__main__":
    import csv
    with open("miniboone_digitized.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["curve", "segment", "m_N_MeV", "d_muN_GeVinv"])
        for name, segs in CURVES.items():
            for i, seg in enumerate(segs):
                for m, d in seg:
                    w.writerow([name, i, f"{m:.6g}", f"{d:.6g}"])
    np.savez("miniboone_digitized.npz",
             **{f"{n}_{i}": s for n, segs in CURVES.items() for i, s in enumerate(segs)})
    for name, segs in CURVES.items():
        tot = sum(len(s) for s in segs)
        print(f"{name:18s}: {len(segs)} segments, {tot} points")
        for s in segs[:3]:
            print(f"    m [{s[:,0].min():7.1f},{s[:,0].max():7.1f}] MeV   "
                  f"d [{s[:,1].min():.2e},{s[:,1].max():.2e}]")
