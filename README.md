# Study on Sound Inventories coded as CLDF Datasets

## Installation Requirements

We assume that you use a freshly created virtual environment in Python (version 3.5 or higher).

```
$ python3 -m venv venv
$ source venv/bin/activate
```

### Install Code Requirements and obtain catalogs

```
$ pip install -r requirements.txt
$ cldfbench catconfig -q
```

### CLDF Datasets

Use `git` to retrieve the transcription datasets.

```
$ git clone https://github.com/cldf-datasets/jipa
$ git clone https://github.com/cldf-datasets/lapsyd
$ git clone https://github.com/cldf-datasets/phoible
```

Install them with `pip`:

```
$ pip install -e jipa
$ pip install -e lapsyd
$ pip install -e phoible
```

### Setup your local version of CLTS data

```
$ git clone https://github.com/cldf-clts/clts.git
```

### Data processing

In order to prepare the data for initial processing, the script `prepare.py` will 

```
$ python prepare.py
```

### Data analysis (python only, preliminary)

```
$ python compare.py
```

### Build plots

It might be necessary to manually installed the required R libraries.

```
$ python plot.py
$ Rscript plot.R
```

