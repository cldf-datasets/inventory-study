"""
Prepare the data for the sound inventories experiment.
"""
from cldfbench import get_dataset
import pycldf
from pyclts import CLTS
from pyclts.inventories import Inventory, Phoneme
from pathlib import Path
from collections import defaultdict
from models import Language, wals_3a
from tqdm import tqdm

def progressbar(function, desc=''):

    return tqdm(function, desc=desc)

# load basic data
clts = CLTS()
bipa = clts.transcriptionsystem_dict['bipa']


def get_cldf_varieties(dataset):
    
    dset = get_dataset(dataset).cldf_reader()
    languages = {row['ID']: row for row in dset.iter_rows('LanguageTable')}
    params = {row['Name']: row for row in dset.iter_rows('ParameterTable')}
    
    varieties = defaultdict(list)
    for row in progressbar(dset.iter_rows('ValueTable'), desc='load values'):
        lid = row['Language_ID']
        varieties[lid] += [row['Value']]
    return languages, params, varieties


def get_phoible_varieties(
        subsets, 
        path=Path.home().joinpath('data', 'datasets', 'cldf', 'cldf-datasets',
            'phoible', 'cldf'),
        ):

    # load phoible data
    phoible = pycldf.Dataset.from_metadata(
            path.joinpath('StructureDataset-metadata.json'))
    
    gcodes = {row['ID']: row for row in phoible.iter_rows('LanguageTable')}
    params = {row['Name']: row for row in phoible.iter_rows('ParameterTable')}
    contributions = {row['ID']: row['Contributor_ID'] for row in
            phoible.iter_rows('contributions.csv')}
    
    td = clts.transcriptiondata_dict['phoible']
    bipa = clts.transcriptionsystem_dict['bipa']
    
    languages = {}
    varieties = defaultdict(list)
    for row in progressbar(phoible.iter_rows('ValueTable'), desc='load values'):
        if contributions[row['Contribution_ID']] in subsets:
            lid = row['Language_ID']+'-'+row['Contribution_ID']
            varieties[lid] += [row['Value']]
            languages[lid] = gcodes[row['Language_ID']]
    return languages, params, varieties



def load_dataset(dataset):
    
    if dataset in ['UZ-PH-GM', 'UPSID']:
        dset_td = clts.transcriptiondata_dict['phoible']
        languages, params, varieties = get_phoible_varieties(dataset)
    else:
        dset_td = clts.transcriptiondata_dict[dataset]
        languages, params, varieties = get_cldf_varieties(dataset)
    
    inventories = {}
    for var, vals in progressbar(varieties.items(), desc='identify inventories'):
        if len(vals) == len(
                [v for v in vals if dset_td.grapheme_map.get(v, '<NA>') != '<NA>']
                ):
            gcode = languages[var]
            lang = Language(
                    var,
                    gcode['Name'],
                    glottocode=gcode['Glottocode'],
                    latitude=gcode['Latitude'],
                    longitude=gcode['Longitude'],
                    family=gcode['Family_Name'],
                    macroarea=gcode['Macroarea'],
                    attributes=gcode
                    )
            sounds = {}
            for v in vals:
                s = dset_td.grapheme_map[v]
                b = bipa[s]
                sounds[str(b)] = Phoneme(
                    grapheme=str(b),
                    grapheme_in_source=v,
                    name=b.name,
                    type=b.type,
                    occs=0,
                    sound=b
                    )
            if lang.latitude:
                inv = Inventory(
                        id=var,
                        sounds=sounds,
                        language=lang,
                        ts=bipa)
                inventories[var] = inv
    
    count = 0
    with open(dataset+'-data.tsv', 'w') as f:
        f.write('\t'.join([ 
            'ID', 'Name', 'Glottocode', 'Family', 'Macroarea',
            'Latitude', 'Longitude', 'Sounds', 'Consonants', 'Vowels',
            'Clusters', 'Diphthongs',
            'Consonantal', 'Vocalic', 'Ratio', 'Phonemes'])+'\n')
        for inv in inventories.values():
            f.write('\t'.join([
                    inv.id, 
                    inv.language.name, 
                    inv.language.glottocode,
                    inv.language.family or '', 
                    inv.language.macroarea or '',
                    str(inv.language.latitude), str(inv.language.longitude),
                    str(len(inv)),
                    str(len(inv.consonants)), 
                    str(len(inv.vowels)),
                    str(len(inv.clusters)), 
                    str(len(inv.diphthongs)),
                    str(len(inv.consonants) + len(inv.clusters)),
                    str(len(inv.vowels) + len(inv.diphthongs)),
                    str(wals_3a(inv)),
                    ' '.join(inv.sounds)
                    ])
                    +'\n')
            count += 1
    print('loaded {0} language varieties for {0}'.format(count, dataset))







for ds in ['UPSID', 'lapsyd', 'jipa', 'UZ-PH-GM']:
    load_dataset(ds)
