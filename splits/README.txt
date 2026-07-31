SRF-UNet dataset split manifests
================================

This directory records the exact image-level splits used by the released
SRF-UNet experiments. Dataset image files are not redistributed here.

Manifest format
---------------

Each train.txt or test.txt file contains:

1. Comment lines beginning with "#". These document the dataset source,
   download links, split protocol, annotation choice, and expected layout.
2. One sample identifier per non-comment line. The identifier is the image
   filename stem after the filename normalization described in that file.

A parser should ignore blank lines and lines beginning with "#".

Expected dataset layout
-----------------------

data/
|-- DRIVE/
|   |-- train/image/
|   |-- train/mask/
|   |-- test/image/
|   `-- test/mask/
|-- STARE/
|   |-- train/image/
|   |-- train/mask/
|   |-- test/image/
|   `-- test/mask/
`-- CHASE_DB1/
    |-- train/image/
    |-- train/mask/
    |-- test/image/
    `-- test/mask/

The current loader pairs image and mask files by normalized filename stem.
Keep each image and its corresponding vessel mask in matching split folders.

Split summary
-------------

DRIVE:
  Official fixed split: 20 training images and 20 test images.

STARE:
  No official train/test split is supplied by STARE.
  Project-specific fixed split: 15 train / 5 test, random seed 2026.

CHASE_DB1:
  No official train/test split is supplied by CHASE_DB1.
  Project-specific subject-level split: 10 train subjects / 4 test subjects,
  giving 20 train images and 8 test images. Left and right eyes from the same
  subject are always kept together. Random seed: 2026.

Reproducibility note
--------------------

Do not silently replace these manifests with another commonly used split.
STARE and CHASE_DB1 have multiple protocols in the literature, so results are
only directly comparable when the same image identifiers and annotation
observer are used.
