# Study on Sound Inventories coded as CLDF Datasets

## Installation Requirements

We assume that you use a freshly created virtual environment in Python (version 3.8 or higher).

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Code Requirements and obtain catalogs

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
cldfbench catconfig -q
```

See `pinned-requirements.txt` for a full list of required packages with specific
versions that is known to work for Python 3.10.


### CLDF Datasets

Use `git` to retrieve the transcription datasets.

```bash
git clone https://github.com/cldf-datasets/jipa
git clone https://github.com/cldf-datasets/lapsyd
git clone https://github.com/cldf-datasets/phoible
```

Install them with `pip`:

```bash
pip install -e jipa
pip install -e lapsyd
pip install -e phoible
```

### Setup your local version of CLTS data

```bash
git clone https://github.com/cldf-clts/clts.git
```

### Data processing

In order to prepare the data for initial processing, include the full data in a
single file, run: 

```bash
python prepare.py
python prepare_fulldata.py
```

### Data analysis (python only, preliminary)

```bash
python compare.py
```

### Build plots

It might be necessary to manually installed the required R libraries.

```bash
python plot.py
Rscript plot.R
```

