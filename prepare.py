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
from pyclts.util import nfd
from pyclts.transcriptionsystem import is_valid_sound

def progressbar(function, desc=''):

    return tqdm(function, desc=desc)

def normalize(grapheme):
    for s, t in [
            ('\u2019', '\u02bc')
            ]:
        grapheme = grapheme.replace(s, t)
    return grapheme



def get_cldf_varieties(dataset):
    
    bipa = CLTS().bipa
    dset = get_dataset(dataset).cldf_reader()
    languages = {row['ID']: row for row in dset.iter_rows('LanguageTable')}
    params = {row['Name']: row for row in dset.iter_rows('ParameterTable')}
    
    varieties = defaultdict(list)
    for row in progressbar(dset.iter_rows('ValueTable'), desc='load values'):
        lid = row['Language_ID']
        varieties[lid] += [nfd(row['Value'])]
    return languages, params, varieties


def get_phoible_varieties(
        subsets, 
        path=Path.home().joinpath('data', 'datasets', 'cldf', 'cldf-datasets',
            'phoible', 'cldf'),
        ):
    bipa = CLTS().bipa
    # load phoible data
    phoible = pycldf.Dataset.from_metadata(
            path.joinpath('StructureDataset-metadata.json'))
    
    gcodes = {row['ID']: row for row in phoible.iter_rows('LanguageTable')}
    params = {row['Name']: row for row in phoible.iter_rows('ParameterTable')}
    contributions = {row['ID']: row['Contributor_ID'] for row in
            phoible.iter_rows('contributions.csv')}
        
    languages = {}
    varieties = defaultdict(list)
    for row in progressbar(phoible.iter_rows('ValueTable'), desc='load values'):
        if contributions[row['Contribution_ID']] in subsets:
            lid = row['Language_ID']+'-'+row['Contribution_ID']
            varieties[lid] += [nfd(row['Value'])]
            languages[lid] = gcodes[row['Language_ID']]
    return languages, params, varieties



def load_dataset(dataset, td=None, clts=None):
    clts = clts or CLTS()

    if not td:
        td = dataset
    
    if dataset in ['UZ-PH-GM', 'UPSID']:
        dset_td = clts.transcriptiondata_dict['phoible']
        languages, params, varieties = get_phoible_varieties(dataset)
    else:
        dset_td = clts.transcriptiondata_dict[td]
        languages, params, varieties = get_cldf_varieties(dataset)
    
    for sound in list(dset_td.grapheme_map):
        dset_td.grapheme_map[normalize(sound)] = dset_td.grapheme_map[sound]
    
    inventories = {}
    count = 0
    soundsD = defaultdict(int)
    exc_file = open('output/excluded.md', 'a')
    exc_file.write('## Dataset {0}\n\n'.format(dataset))
    for var, vals in progressbar(varieties.items(), desc='identify inventories'):
        for sound in vals:
            if sound not in dset_td.grapheme_map:
                bsound = bipa[sound]
                if bsound.type != 'unknownsound' and is_valid_sound(bsound, bipa):
                    dset_td.grapheme_map[sound] = bipa[sound].s
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
                    family=gcode.get('Family_Name', gcode.get('Family')),
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
        else:
            for sound in vals:
                if sound not in dset_td.grapheme_map or dset_td.grapheme_map[sound] == '<NA>':
                    soundsD[sound] += 1
            exc_file.write('### Variety {0} ({1})\n\n'.format(
                var, gcode['Glottocode']))
            exc_file.write('Sound in Source | BIPA \n')
            exc_file.write('--- | --- \n')
            for sound in vals:
                exc_file.write(sound + ' | ' + dset_td.grapheme_map.get(
                    sound, '<NA>') +' \n')
            count += 1
    exc_file.write('\n\n')
    exc_file.close()
    print('[i] excluded {0} inventories for {1}'.format(count, dataset))
    print('Problematic sounds: {0}'.format(len(soundsD)))
    for s, count in sorted(soundsD.items(), key=lambda x: x[1]):
        print('{0:8} \t| {1}'.format(s, count))
    input()
    
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



# load basic data
clts = CLTS()
bipa = clts.transcriptionsystem_dict['bipa']


with open('output/excluded.md', 'w') as f:
    f.write('# Excluded Varieties\n\n')

print('eurasianinventories')
load_dataset('eurasianinventories', td='eurasian')
for ds in ['UPSID', 'lapsyd', 'jipa', 'UZ-PH-GM']:
    print(ds)
    load_dataset(ds)

