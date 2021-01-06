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
import json
import pybtex
from pycldf.sources import Source
from pylatexenc.latex2text import LatexNodes2Text

def progressbar(function, desc=''):

    return tqdm(function, desc=desc)

def normalize(grapheme):
    for s, t in [
            ('\u2019', '\u02bc')
            ]:
        grapheme = grapheme.replace(s, t)
    return grapheme



def get_cldf_varieties(dataset):
    """
    Load a generic CLDF dataset.
    """
    bipa = CLTS().bipa
    dset_ = get_dataset(dataset)
    dset = dset_.cldf_reader()
    try:
        bib = {source.id: source for source in dset_.cldf_dir.read_bib()}
    except:
        bib = {}
    dset = get_dataset(dataset).cldf_reader()
    languages = {row['ID']: row for row in dset.iter_rows('LanguageTable')}
    params = {row['Name']: row for row in dset.iter_rows('ParameterTable')}
    varieties = defaultdict(list)
    sources = defaultdict(set)
    for row in progressbar(dset.iter_rows('ValueTable'), desc='load values'):
        lid = row['Language_ID']
        source = row['Source'][0] if row['Source'] else ''
        varieties[lid] += [nfd(row['Value'])]
        sources[lid].add(source)
    return languages, params, varieties, sources, bib


def get_phoible_varieties(
        subsets, 
        path=Path.home().joinpath('data', 'datasets', 'cldf', 'cldf-datasets',
            'phoible', 'cldf'),
        ):
    """
    Load phoible data (currently not in generic CLDF).
    """
    bipa = CLTS().bipa
    phoible = pycldf.Dataset.from_metadata(
            path.joinpath('StructureDataset-metadata.json'))
    bib = pybtex.database.parse_string(
            open(path.joinpath('sources.bib').as_posix()).read(), bib_format='bibtex')
    bib_ = [Source.from_entry(k, e) for k, e in bib.entries.items()]
    bib = {source.id: source for source in bib_}
    gcodes = {row['ID']: row for row in phoible.iter_rows('LanguageTable')}
    params = {row['Name']: row for row in phoible.iter_rows('ParameterTable')}
    contributions = {row['ID']: row['Contributor_ID'] for row in
            phoible.iter_rows('contributions.csv')}
    languages = {}
    varieties = defaultdict(list)
    sources = defaultdict(set)
    for row in progressbar(phoible.iter_rows('ValueTable'), desc='load values'):
        if contributions[row['Contribution_ID']] in subsets:
            lid = row['Language_ID']+'-'+row['Contribution_ID']
            varieties[lid] += [nfd(row['Value'])]
            languages[lid] = gcodes[row['Language_ID']]
            source = row['Source'][0] if row['Source'] else ''
            sources[lid].add(source)
    return languages, params, varieties, sources, bib


def style_source(sources, bib):
    source = sources.pop()
    if source in bib:
        tmp = {}
        for k, v in bib[source].items():
            tmp[k.lower()] = LatexNodes2Text().latex_to_text(v)
        return '{0} ({1}): {2} [{3}]'.format(
                tmp.get('author', '?'),
                tmp.get('year', '?'),
                tmp.get('title', '?'),
                source)
    elif source:
        print('Missing source {0}'.format(source))
        return source
    return ''



def load_dataset(dataset, td=None, clts=None, dump=defaultdict(list)):
    clts = clts or CLTS()

    if not td:
        td = dataset

    if dataset in ['UZ-PH-GM', 'UPSID']:
        dset_td = clts.transcriptiondata_dict['phoible']
        languages, params, varieties, sources, bib = get_phoible_varieties(dataset)
    else:
        dset_td = clts.transcriptiondata_dict[td]
        languages, params, varieties, sources, bib = get_cldf_varieties(dataset)

    for sound in list(dset_td.grapheme_map):
        dset_td.grapheme_map[normalize(sound)] = dset_td.grapheme_map[sound]

    inventories = {}
    missing_gcodes = 0
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
        gcode = languages[var]

        if len(vals) == len(
                [v for v in vals if dset_td.grapheme_map.get(v, '<NA>') != '<NA>']
                ):
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
                if b.type in ['vowel', 'consonant', 'diphthong', 'cluster']:
                    sounds[str(b)] = Phoneme(
                        grapheme=str(b),
                        grapheme_in_source=v,
                        name=b.name,
                        type=b.type,
                        occs=0,
                        sound=b
                        )
                    dump['bipa-'+s] = b.name

            if lang.glottocode:
                inv = Inventory(
                        id=var,
                        sounds=sounds,
                        language=lang,
                        ts=bipa)
                inventories[var] = inv
                dump[gcode['Glottocode']] += [
                        {
                            'ID': var,
                            'Dataset': dataset,
                            'Name': gcode['Name'],
                            'Source': style_source(sources[var], bib),
                            'CLTS': {
                                sound.grapheme: sound.grapheme_in_source for sound in sounds.values()},
                            'Sounds': vals}]
            else:
                missing_gcodes += 1
            
        else:
            for sound in vals:
                if sound not in dset_td.grapheme_map or dset_td.grapheme_map[sound] == '<NA>':
                    soundsD[sound] += 1
            exc_file.write('### Variety {0} ({1}, {2})\n\n'.format(
                var, gcode['Glottocode'], ' '.join(sources[var])))
            exc_file.write('Sound in Source | BIPA \n')
            exc_file.write('--- | --- \n')
            for sound in vals:
                exc_file.write(sound + ' | ' + dset_td.grapheme_map.get(
                    sound, '?') +' \n')
            count += 1
    exc_file.write('\n\n')
    exc_file.close()
    print('[i] excluded {0} inventories for {1}'.format(count, dataset))
    print('missing gcodes: {0}'.format(missing_gcodes))
    print('Problematic sounds: {0}'.format(len(soundsD)))
    for s, count in sorted(soundsD.items(), key=lambda x: x[1]):
        print('{0:8} \t| {1}'.format(s, count))
    
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
                    str(inv.language.latitude or ''),
                    str(inv.language.longitude or ''),
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
    return dump

# load basic data
clts = CLTS()
bipa = clts.transcriptionsystem_dict['bipa']


with open('output/excluded.md', 'w') as f:
    f.write('# Excluded Varieties\n\n')

#print('eurasianinventories')
#dump = load_dataset('eurasianinventories', td='eurasian')
dump = defaultdict(list)
for ds in ['jipa', 'UPSID', 'lapsyd', 'UZ-PH-GM']:
    print(ds)
    dump = load_dataset(ds, dump=dump)
with open('app/data.js', 'w') as f:
    f.write('var DATA = '+json.dumps(dump, indent=2)+';\n')

