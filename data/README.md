# Dataset layout

Obtain the datasets from their original providers under their respective terms.
Datasets are not redistributed in this repository.

```
data/
  PU/PaviaU.mat       # key: paviaU, shape (610, 340, 103)
  PU/PaviaU_gt.mat    # key: paviaU_gt, shape (610, 340), 0 = unlabeled
  HUS/dfc2013.mat     # keys: data_hs (349, 1905, 144), train, test
```

`train` and `test` in the Houston file are disjoint class-label maps. Random
splits use their sum; the official spatially disjoint protocol keeps them separate.
Class identifiers are 1..15 for Houston and 1..9 for Pavia University.
`code/common.py` defines the exact per-class training/validation budgets.
The repository defaults to its own root. Set `ATOF_ROOT` to another directory
containing `data/` and `results/` to reuse an existing local experiment workspace.
