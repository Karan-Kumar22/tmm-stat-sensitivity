# Early DarkNews scans

Standalone (m4, mu_tr) scans over a 20x20 grid, cryostat + dirt, from
before the grid-campaign machinery in `generation/` existed. Only the
scripts whose output feeds kept results are archived; superseded by
`generation/` for all current results.

| script | role |
|---|---|
| scan_darknews_2.py    | the scan behind the tmm-vs-ncqe comparison figure (noHC=False rows): mass-dependent Umu4 mixing (get_Umu4 windows), resumable partial CSVs |
| scan_darknews_fill.py | patch: the two masses (0.1227, 0.1622 GeV) that fell in a gap of scan_darknews_2's Umu4 windows |

Other exploratory configurations (HC term off, pure dipole with
Umu4 = 0) existed as sibling scripts and remain on the Nevis cluster at
/nevis/riverside/data/karan/TMM_Analysis/TMM_Karan/scan/.

All ran under the DarkNews virtualenv (~/TMM_new/.venv-tmm).
