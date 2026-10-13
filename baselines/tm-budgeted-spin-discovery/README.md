# Budgeted spin-discovery baselines

Three JSON-lines strategies provide reference scores for the 419-candidate final pool. Each of ten conditions supplies 50 initial labels and permits 100 unique new queries.

| Strategy | Mean discoveries |
| --- | ---: |
| Uniform random | 13.2 |
| Static absolute-mean ranking | 26.8 |
| Adaptive interval probability | 32.4 |

The two model strategies fit ExtraTreesRegressor with 128 trees, minimum leaf size 2, all features, the supplied seed and one numerical thread. The adaptive strategy refits every ten new labels. Tree spread supplies an acquisition heuristic, without an uncertainty calibration claim. Strategies fit only their condition's initial labels and authorized feedback; each condition starts in a fresh process.

Submit the selected directory containing solution/. Python 3.11, numpy 2.3.2, scipy 1.17.1 and scikit-learn 1.7.2 were used for the recorded offline Linux runs. Those runs used a compatible CPU Docker image and a public-API VerifyContext test double. Integration with the operator's actual SDK and image remains unverified.

Discoveries satisfy abs(adiabatic signed gap)<5 kcal/mol. This is a reference-DFT screen, with ten initial conditions on one dataset. Development and final pools contain 190 and 419 candidates and have no shared parents with P3. Source targets are public and were inspected during construction. Meyer, Chu and Kulik, DOI 10.1088/2632-2153/ad9f22; many_body_ml commit 27467b5d3d85fa42263d746d514461fa009b3c91. See NOTICE.txt for the MIT license and attribution.
