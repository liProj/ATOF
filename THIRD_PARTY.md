# External resources and attribution

- DINOv3: https://github.com/facebookresearch/dinov3 . The main visual branch uses
  `vit_large_patch16_dinov3.sat493m` through timm. Obtain weights according to their
  upstream access and license terms; weights are not redistributed here.
- timm: https://github.com/huggingface/pytorch-image-models . Model construction
  and intermediate feature extraction use this library.
- LDMAN: https://github.com/sg10216/LDMAN . Optional comparison only; source is
  not bundled. See `third_party/README.md`.
- Centered kernel alignment follows Cortes, Mohri, and Rostamizadeh,
  “Algorithms for Learning Kernels Based on Centered Alignment,” JMLR 13,
  795–828, 2012.
- Houston 2013 and Pavia University remain governed by the data providers' terms.

Public availability of this repository does not replace the upstream licenses.
No additional redistribution license is assigned to third-party resources.
