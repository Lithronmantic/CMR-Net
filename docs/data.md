# Data

The Intel robotic welding dataset is **not** distributed with this repository.
Obtain it separately and point the config at your local copy — do not hard-code
an absolute path.

## Configuring the path

```yaml
# configs/default.yaml
data:
  real:
    root: ${oc.env:CMR_DATA_ROOT, data/intel_robotic_welding_dataset}
    label_mode: suffix12
```

```bash
export CMR_DATA_ROOT=/path/to/intel_robotic_welding_dataset
```

## Expected contents

The loader (`src/cmr_net/data/dataset.py`) expects, per sample, the synchronized
process time series, audio spectrogram, and video clip, with a fine-grained
defect label (`label_mode: suffix12`).

## Labels and families

12 fine-grained defect classes are aggregated into 6 engineering families
(see the paper appendix "Data, Preprocessing, and Family Definition"):

`good` · `penetration depth` · `porosity` · `geometry profile` ·
`surface distortion` · `cracking`.

The family aggregation is a modeling choice for certificate evaluation; it is
not a strict metallurgical taxonomy.
