# ROQ (Read Overlapping Quality)

ROQ is an XGBoost-based read alignment quality metric that predicts how well
a mapped read's reported position overlaps its true origin. It is designed
as a drop-in complement to MAPQ, and is described in the accompanying
manuscript.

## Directory structure

```
ROQ/
├── README.md
├── LICENSE
├── environment.yml        Minimal environment: run ROQ on your own BAM
├── bin/
│   ├── ExtractFeatures.pl Feature extraction (BAM/SAM/BED -> feature TSV)
│   └── predROQ.py         Applies the trained model, tags reads with ZQ
├── example/                Small example input + expected output
├── model/
│   ├── model.pkl           Final trained model (pickle)
│   └── model.json          Same model, XGBoost's native format (see below)
└── evaluation/              Scripts used to produce the manuscript's
                             tables and figures -- see evaluation/README.md.
                             Not needed to just run ROQ.
```

<br>
<br>

## Quick start: score your own BAM
> Requires [Git LFS](https://git-lfs.com) to download the model files — see [Model files](#model-files).

```bash
# 1. Extract features
perl bin/ExtractFeatures.pl input.bam > features.tsv

# 2. Score reads and tag the BAM
python bin/predROQ.py \
  --feature-set portable_no_scores \
  -m model/model.pkl \
  -i features.tsv \
  -ib input.bam \
  -head 0 \
  -o output_dir/
```

The tagged BAM is written to `output_dir/output.bam`, with a `ZQ:f:<value>`
tag on every read that had a matching row in `features.tsv`. `ZQ` ranges
from 0 to 42, on the same scale as MAPQ:

```
ZQ = 42 / (1 + exp(-10 * (p - 0.5)))
```

where `p` is the model's raw predicted overlap probability. We use `ZQ`
rather than `RQ` because PacBio HiFi/CCS BAMs already carry a native
`RQ:f:` tag (predicted read quality from CCS); reusing that tag would
silently collide with it. `Z*` tags are reserved by the SAM spec for
local/custom use.

See `example/` for a worked input/output pair you can use to sanity-check
your own install.

### Feature set

The released model uses 9 features (`--feature-set portable_no_scores`):
`MAPQ, MISMATCHES, GAP_OPENS, GAP_EXT, EDIT_DIST, INSERT_SIZE,
READ_GC_CONT, N_LOW_QUALITY_BASE, AVG_QUALITY_BASE_SCORE`.

`--feature-set full` (13 features, adding aligner-score fields) is 
also accepted by both scripts, but is for evaluation/ use only and 
is not compatible with the released model.

### Model files

`model/model.pkl` is a pickled `xgboost.XGBRegressor` (9 features, 24,000
boosted rounds). `model/model.json` is the same model exported with
XGBoost's own `Booster.save_model()`, kept alongside the pickle as a
version-compatibility fallback — `pickle` can break across major XGBoost
versions, `save_model()`'s format is guaranteed forward-compatible. Load
whichever fits your code; `predROQ.py` uses the pickle by default.

> **Note:** Both model files are stored with [Git LFS](https://git-lfs.com).
> A plain `git clone` without Git LFS (common on Linux servers) or the
> "Download ZIP" button downloads only small pointer files (~134 bytes),
> and `predROQ.py` will fail when loading the model. To fetch the actual files:
>
> ```bash
> conda install -c conda-forge git-lfs   # or: sudo apt install git-lfs / brew install git-lfs
> git lfs install
> git lfs pull                           # run inside the ROQ directory
> ls -lh model/                          # model.pkl ≈ 99 MB, model.json ≈ 140 MB
> ```


## Reproducing the manuscript's results

See [`evaluation/README.md`](evaluation/README.md).

## License

MIT. See `LICENSE`.

## Contact

For questions or issues, please contact: 
- ibclab.kr@gmail.com
- qkrskdud0805@gmail.com
