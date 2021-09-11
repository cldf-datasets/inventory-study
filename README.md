# Study on Sound Inventories coded as CLDF Datasets

## Installation Requirements

We assume that you use a freshly created virtual environment in Python (version 3.5 or higher).

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

### Install Code Requirements

```
$ pip install -r requirements.txt
```

### Set up your version of the CLTS data

```
$ git clone https://github.com/cldf-clts/clts.git
$ cldfbench catconfig
```

You will be prompted to provide the current location of the `clts` directory which contains the data you need to run the code related to the CLTS transcription system reference catalog.


### Data processing

In order to prepare the data for initial processing, the script `prepare.py` will 

```
$ python prepare.py
```

### Data analysis (python only, preliminary)
```
$ python compare.py
```
