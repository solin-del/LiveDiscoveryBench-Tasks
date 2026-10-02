# Documented 116-channel archived feature input

## Immutable scientific sources

Method-table data and generated VSS geometries: Duan et al., Zenodo 7350958 v2, DOI https://doi.org/10.5281/zenodo.7350958, CC BY 4.0. The VSS structures were assembled with molSimplify; CSD-derived coordinates are excluded. Scientific methods: https://doi.org/10.1038/s43588-022-00384-0 and https://arxiv.org/pdf/2207.10747 (Methods).

Density arrays and normalization metadata: Garrison and Kulik, Zenodo 17468690 v1, DOI https://doi.org/10.5281/zenodo.17468690, CC BY 4.0. Archive SHA-256: bfd43e3fbd5db3611d28f5a7bc3f59894fcbe8a1fabfecebf83f3a603463081f. Required feature arrays, normalizers and DF Featurization.ipynb match the Git blobs at https://github.com/hjkgrp/hfx_rec/tree/7bd4e453513449be1e34c064e61a601528ba4bc9. This does not assert that every archive file matches later repository text. Source generation code references df_class.py and df_utils.py at https://github.com/hjkgrp/dfa_recommender/tree/e76c9b3ac3ee3a97fd339fdd9728fcf3aaa35e77. No original source model is executed or redistributed.

## Physical channels and recovery

At the source B3LYP high-spin optimized geometry, the difference of PBE0 HS and LS density matrices is fitted separately for alpha and beta spin using the def2-TZVP JKFIT auxiliary basis, three-center integrals and inverse auxiliary Coulomb metric. The source metric inverse uses power(-1.0, 1.e-14). Squared fitting coefficients are summed within each auxiliary shell. Preserve the source atom-local shell order, zero-pad each spin block to 58 values, then concatenate alpha channels 0-57 and beta channels 58-115. Channel numbers are not universal angular-momentum labels across elements. The power spectrum drops coefficient signs and is not an invertible full density representation; no atom-charge units are assigned.

The source PBE0 VSS array is 348 x 65 x 117: 116 standardized numeric channels plus a source network-group code. The source groups are X,H,C,N,O,F,Cr,Mn,Fe,Co, with P mapped to N, S to O and Cl to F. For active atom rows recover raw = normalized * std + mean from the source PBE0 standard_dict; never add means to padding rows. Drop the group-code column and all nonexistent-atom padding rows. Preserve actual element symbols separately. Raw power values are nonnegative except roundoff within 1e-8. This procedure is not truncation or recovery of the original undefined 181-channel descriptor.

The source notebook standardizes using all retained VSS density rows, including source test rows. That transform is inverted before export. All new model scaling is fitted on training or training-fold labels only. This is recovery of archived numbers, not an independent fresh quantum calculation.

## Cohort and association limits

The source feature notebook requires a finite cleaned PBE HFX25 gap. Its curve cleaner iteratively removes a point when omitting it changes the interpolating VSSE-versus-HFX spline by more than 5 kcal/mol. There are 401 raw finite PBE25 state pairs, 382 after cleaning, and 348 with archived PBE0 densities. All 348 occur in the cleaned set. The task additionally requires a finite reference and all 45 fixed method gaps, then applies the declared cross-task formula exclusions and original train/test flags. This is a conditional source-filtered cohort; records are not selected by this recommender's final prediction error.

Source identifiers, atom counts, network-group-code sequences and zero padding were checked against same-named XYZ records for all VSS inputs. Shared codes cannot independently certify P/N, S/O or Cl/F ordering within a group, permutations of same-element atoms, or every original wavefunction coordinate. The sole archived CSD example independently matches actual element order, geometry and two-state metadata, but remains construction-only; its coordinates and wavefunctions are not included and that evidence is not extrapolated to every VSS record.

## Frozen recommender and reconstruction

For actual elements H,C,N,O,F,P,S,Cl,Br,I,Cr,Mn,Fe,Co concatenate the atom count, per-channel mean and population standard deviation (ddof=0), zeroing absent elements. Append the same 77-dimensional structure/charge descriptor; the density_mean_std view has 3339 dimensions. The supplied raw116 channels, geometry, charge and comparison multiplicities are the allowed inputs. Structure-only geometry_charge has 77 dimensions.

The chosen multi-output ExtraTrees model has 256 trees, seed 20260930, n_jobs=1, min_samples_leaf=1, max_features=0.5. Targets are each method's absolute gap error against the reference, standardized separately on training only. Undo target scaling at inference, then choose the first minimum in the fixed method registry. Five-fold GroupKFold by full elemental formula selected the model and view before this v2 cohort was scored. The public recipe rebuilds the frozen parameters from the 117 public training rows; it neither searches parameters nor opens final data.

Run `python recipe/p5_defined_rebuild.py --training /path/to/task/input/data/training.json --recipe recipe/frozen_recipe.json --output /new/output`. Use Python with numpy 2.3.2, scipy 1.17.1, scikit-learn 1.7.2 and threadpoolctl 3.7.0, with one numerical thread. Pure NPZ inference requires no sklearn object, source pickle or original feature-generation code. Reconstruction has been verified by zero numerical-array difference from the frozen artifacts.

Two PBE0 state SCFs and prior B3LYP geometry preparation are required to obtain these electronic inputs. All inputs here are archived; full source settings and wavefunctions are unavailable. Full fresh SCF/density-fitting reproduction and end-to-end timing or cost advantage are unmeasured. DLPNO-CCSD(T0) is an approximate archived reference, not exact or experimental truth. The original 181-channel recipe remains unresolved.
