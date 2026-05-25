# Data directory

This repository intentionally does not include the TEP or FCC datasets.

Place data manually using one subfolder per class, for example:

```text
data/tep/class_01/run_0001.csv
data/tep/class_01/run_0002.csv
data/tep/class_02/run_0001.csv
```

Each CSV should contain a binary alarm activity matrix with one row per time step and one column per alarm tag. Optional time columns such as `Minutes`, `time`, `timestamp`, or `t` are detected automatically.
