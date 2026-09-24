Controlled ablation archive
Machine: ISS node1
GPU: NVIDIA GeForce RTX 4090

Cases:
  pglib_opf_case162_ieee_dtc
  pglib_opf_case300_ieee

Epochs: 10000
Seeds: 1,2,3,4,5

Per case:
Projection study:
  none       + nominal
  voltage    + nominal
  generation + nominal
  full       + nominal

Weight study:
  full + equal
  full + inequality
  full + physics
  full + objective

Expected checkpoints per case: 40
Total checkpoints: 80
