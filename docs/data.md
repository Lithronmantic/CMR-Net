# Data

The Intel Robotic Welding Multimodal Dataset is not distributed with this repository. Obtain the dataset separately and set its local path through `CMR_DATA_ROOT`.

```bash
export CMR_DATA_ROOT=/path/to/intel_robotic_welding_dataset
```

The configuration file is `configs/welding_inference.yaml`.

Each sample contains a synchronized process time series, audio spectrogram, video clip, fine-grained defect label, process summary, and context vector.

The input dimensions are:

| Input | Shape |
|---|---|
| Process sequence | `32 x 6` |
| Audio spectrogram | `48 x 64` |
| Video clip | `8 x 3 x 32 x 32` |
| Context vector | `7` |

The 12 fine-grained defect classes are aggregated into six engineering families:

- `good`
- `penetration depth`
- `porosity`
- `geometry profile`
- `surface distortion`
- `cracking`

The family aggregation defines the certificate-evaluation units and is not a metallurgical taxonomy.
