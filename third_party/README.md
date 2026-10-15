# Optional LDMAN comparison

The ATOF pipeline does not require LDMAN. To run the comparison, obtain the
authors' implementation from https://github.com/sg10216/LDMAN and place its
modules under `third_party/LDMAN/`, subject to the upstream terms.
This repository contains our evaluation runner, not a redistributed copy of
the third-party network. Check that upstream imports (including `models`) resolve
in your environment before running `code/run_ldman.py`.

The archived matched baseline used `PRE=nowhiten`, 150 epochs, and best validation
OA checkpoint selection. The official Houston split used 60 epochs.
Original published comparison values and these matched runs are different sources.
