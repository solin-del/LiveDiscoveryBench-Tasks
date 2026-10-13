# DFA selection baseline

This baseline chooses one of 45 archived DFA methods from a complex's geometry, charge and 116 PBE0 density-difference channels. The frozen density model scores 2.6460461241067943 kcal/mol MAE on 58 final VSS records. A training-selected fixed method scores 3.163990678933186; geometry alone scores 3.0180903363003124. The oracle MAE is 0.4183947155527795.

The model predicts each method's absolute reference error and selects the first minimum in the fixed registry. Five-fold formula-group cross-validation on 117 training records selected element-wise density means and population standard deviations, 256 ExtraTrees, min_samples_leaf=1, max_features=0.5 and seed 20260930. The geometry comparator uses min_samples_leaf=8 and the same remaining settings. Target scaling is fitted on training data. Frozen inference uses numeric NPZ arrays.

Run the public training recipe:

```text
python recipe/p5_defined_rebuild.py --training /path/to/task/input/data/training.json --recipe recipe/frozen_recipe.json --output /new/output
```

The pinned dependencies are NumPy 2.3.2, SciPy 1.17.1, scikit-learn 1.7.2 and threadpoolctl 3.7.0, with one numerical thread. The recipe fits the declared configurations from training only. Submit the directory containing solution/. training_best_fixed/ selects m06-l_hfx_30. The archived reconstruction audit reports zero differences in every exported numeric array.

FEATURE_RECIPE.md explains the 58 alpha plus 58 beta channels, inversion of source scaling, pooling, source filtering and geometry association. This documented 116-channel task replaces the original 181-channel representation and uses an independently trained model. Obtaining electronic inputs requires two PBE0 state SCFs and a B3LYP-prepared high-spin geometry. Full fresh density reconstruction and end-to-end cost savings remain unmeasured.

The cohort retains source convergence, spin-contamination and HFX-curve screening, requires complete method/reference values and archived density features, and excludes P2/P3 formulas. Training and final formulas are disjoint; P1/P5 final records can overlap. The two comparison states follow Duan et al., including the d4 singlet/quintet convention. Shared source atom-group codes limit independent verification of within-group atom order. Published targets were inspected during construction. DLPNO-CCSD(T0) is an approximate reference.

Paired formula-cluster bootstrap intervals for the density improvement span zero: [-0.39499, 1.50759] kcal/mol versus the fixed method and [-0.18793, 1.05977] versus geometry. This cohort does not establish superiority.

measured/linux_report.json records compatible-container inference, scoring, eight invalid responses, replay, an isolation-boundary probe and an independent error/oracle/regret fixture. Those checks used a public-API VerifyContext double. The operator SDK, image and root/chroot/setuid/syscall capabilities require verification. The submission runtime enforces 1800 seconds, 4 GiB address space, offline execution and one numerical thread.

Method/geometry data: [Duan et al., Zenodo 7350958 v2](https://doi.org/10.5281/zenodo.7350958). Density data: [Garrison and Kulik, Zenodo 17468690 v1](https://doi.org/10.5281/zenodo.17468690). Both archives specify CC BY 4.0. NOTICE.txt records attribution and data modifications. The [pinned density notebook](https://github.com/hjkgrp/hfx_rec/tree/7bd4e453513449be1e34c064e61a601528ba4bc9) supplies the source feature recipe.

Training hash checks normalize CRLF to LF. Parsed inputs, frozen configurations and numeric model arrays retain their committed values.
