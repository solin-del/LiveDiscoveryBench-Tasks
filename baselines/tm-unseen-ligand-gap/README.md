# Unseen-ligand spin-gap baselines

These baselines use 956 training records, 235 development records and 127 final records. Final ligands are absent from training and development, and P2 parents are excluded.

| Baseline | Final MAE, kcal/mol |
| --- | ---: |
| Three-seed RAC MLP | 5.263004864168261 |
| RAC KRR | 5.846056702019194 |
| Training-mean constant | 19.356138664589047 |

rac_mlp_protocol.json defines four candidate configurations, three seeds, training-only standardization and a 500-iteration cap. Development MAE selects the configuration. The selected runs reached that cap. Supplied NPZ files store numeric parameters loaded with allow_pickle=False.

To retrain with the supplied training/development data:

```text
python train_baseline.py --input /path/to/task/input --output /new/output --baseline rac_mlp
```

Use rac_krr to select the kernel baseline. Final references are excluded from the training recipe. Submit the baseline directory containing solution/ to run its persistent JSON-lines entry.

Recorded scores come from offline Linux inference with Python 3.11, numpy 2.3.2, scipy 1.17.1, scikit-learn 1.7.2 and one numerical thread. Retraining may vary across numerical libraries or platforms; the supplied inference assets define these reference scores. The recorded runs used a compatible CPU Docker image and a public-API VerifyContext test double. Integration with the operator's actual SDK and image remains unverified.

Ligand equivalence uses element adjacency and charge, ignoring bond order and stereochemistry. Source targets are public and were inspected during construction; this is not an uncontaminated holdout. The paper's 4.00 kcal/mol result uses different training conditions. Attribution: Meyer, Chu and Kulik, DOI 10.1088/2632-2153/ad9f22; many_body_ml commit 27467b5d3d85fa42263d746d514461fa009b3c91. See NOTICE.txt for the MIT license and attribution.
