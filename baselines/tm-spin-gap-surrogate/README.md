# Geometry baselines for archived B3LYP vertical spin gaps

These baselines predict archived B3LYP/def2-TZVP E_HS(R_HS)-E_LS(R_HS) from the supplied B3LYP high-spin optimized geometries (LANL2DZ on the metal, 6-31G* on other atoms), charge and comparison multiplicities. The task contains 173 training, 45 development and 103 final author-generated VSS records, in 173/44/101 disjoint elemental-formula groups.

The descriptor combines element counts, metal identity, total charge, comparison multiplicities, metal radial statistics by element and a zero-padded Coulomb-matrix eigenvalue spectrum. Standardization uses training records only. Development MAE selects configurations and the reference family. Frozen NPZ files hold numerical arrays; tree inference casts inputs to float32 to match sklearn.

| Model | Development MAE | Final Linux MAE (kcal/mol) |
| --- | ---: | ---: |
| Three-seed geometry MLP (reference anchor) | 7.548646368247669 | 7.793858458391523 |
| Geometry ExtraTrees | 7.556141943441921 | 5.077471566282976 |
| Geometry KRR | 7.991817031937034 | 9.51154966312888 |
| Training median (bottom anchor) | see rebuild report | 15.373565448535388 |

The MLP has hidden layers (256,128), alpha=0.01 and seeds 61041/61042/61043. One member reaches the 500-iteration cap. Trees have the lowest final MAE; the MLP remains the reference because it had the lowest development MAE. These independent geometry models are not reproductions of published SLATM, SOAP, MACE or 3DMol networks.

To rebuild from public inputs, use Python with numpy 2.3.2, scipy 1.17.1, scikit-learn 1.7.2 and threadpoolctl 3.7.0:

```sh
python recipe/rebuild.py \
  --training /path/to/task/input/data/training.json \
  --development /path/to/task/input/data/development.json \
  --output /new/output
```

The recipe uses one numerical thread, reads training/development only and requires a new output directory. Input SHA-256 guards normalize CRLF to LF. Earlier reconstruction audits report zero differences in all frozen numerical arrays; measured inference differences allow floating-point roundoff. Submit the directory containing `solution/` for the selected model.

The comparison states follow [Duan et al.'s source Methods](https://doi.org/10.1038/s43588-022-00384-0): d3/d7=(2,4), d4/d6=(1,5), d5=(2,6), for Cr/Mn/Fe/Co in oxidation states II/III. The source d4 singlet/quintet convention omits the intermediate triplet. Singlet single points are restricted and other single points unrestricted. Source HS metadata, charge and electron parity were checked without target signs. Original electronic outputs are unavailable, so source SCF convergence and numerical settings have not been independently recomputed.

Final targets contain 103 negative gaps, with no positive or zero gaps; training has 166/7/0 and development 43/2/0. Two positive source-test records were excluded by the predeclared cross-task formula rule. Continuous-gap MAE measures this task; the final set cannot assess balanced spin-order classification. Formula grouping does not prove full molecular-identity isolation, and final chemical records may overlap with the system-specific DFA task. Public targets were inspected during benchmark construction. Geometry preparation requires prior quantum calculations, whose cost is outside the measured inference task. Generated VSS geometries and B3LYP targets differ from the proposed crystal-geometry B3LYP* task.

`measured/linux_report.json` and `measured/results/` record offline Linux inference/scoring for the three models and training median, six malformed responses, an isolation-boundary probe and deterministic tree replay. Those checks used a CPU compatibility container and a public-API VerifyContext test double. The runtime needs Linux x86-64 with root, chroot, setuid and seccomp capabilities; inference is offline and single-process, with 1800-second wall and 4-GiB address-space limits. Execution against the actual operator SDK/image remains a merge validation item. Historical measurements are preserved; only the explanation in `measured/scorer_equivalence.json` was edited for clarity. The readability regression is recorded outside this baseline in `audit/verification.json`.

Scientific data: [Duan et al., Zenodo 7350958 v2](https://doi.org/10.5281/zenodo.7350958), CC BY 4.0. `NOTICE.txt` preserves attribution and identifies solin-del's transformations. Only author-generated VSS structures are included. Data notices do not relicense third-party code.
