from csvw.dsv import UnicodeDictReader
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from matplotlib import pyplot as plt
from tabulate import tabulate
from statistics import median, mean
from itertools import combinations, product
from pyclts.inventories import Inventory
from pyclts import CLTS
from tqdm import tqdm as progressbar

def to_dict(path, parameters):
    with UnicodeDictReader(path+'-data.tsv', delimiter='\t') as reader:
        data = {row['ID']: row for row in reader}
    gcodes = defaultdict(list)
    for row in data.values():
        gcodes[row['Glottocode']] += [row['ID']]
    
    # get the median per feature per glottocode
    values = {k: {p: '' for p in parameters} for k in gcodes}
    for gcode, varieties in gcodes.items():
        for p in parameters:
            param = []
            for variety in varieties:
                if data[variety].get(p, ''):
                    param += [float(data[variety][p])]
            if param:
                values[gcode][p] = median(param)
    return data, gcodes, values


def inventories(path, ts):
    with UnicodeDictReader(path+'-data.tsv', delimiter='\t') as reader:
        data = {row['ID']: row for row in reader}
    gcodes = defaultdict(list)
    for row in data.values():
        gcodes[row['Glottocode']] += [Inventory.from_list(
            *row['Phonemes'].split(' '),
            language=row['ID'],
            ts=ts)]
    return gcodes


def deltas(lstA, lstB):
    score = 0
    for a, b in zip(lstA, lstB):
        score += abs(a-b)
    return score / len(lstA)


def compare_inventories(dctA, dctB, aspects, similarity='strict'):
    scores = []
    for code in dctB:
        if code in dctA:
            invsA, invsB = dctA[code], dctB[code]
            score = []
            for invA, invB in product(invsA, invsB):
                if similarity == 'strict':
                    score += [invA.strict_similarity(invB, aspects=aspects)]
                else:
                    score += [invA.approximate_similarity(invB, aspects=aspects)]
            score = mean(score)
            scores += [score]
    return mean(scores)

bipa = CLTS().bipa

parameters = ['Sounds', 'Consonantal', 'Vocalic']

(
        (jpa_data, jpa_codes, jpa),
        (lps_data, lps_codes, lps),
        (ups_data, ups_codes, ups),
        (stm_data, stm_codes, stm)
        ) = [to_dict(ds, parameters) for ds in [
            'jipa', 'lapsyd', 'UPSID', 'UZ-PH-GM']]


jpaD, lpsD, upsD, stmD = (
        inventories('jipa', bipa),
        inventories('lapsyd', bipa),
        inventories('UPSID', bipa),
        inventories('UZ-PH-GM', bipa))

idxs = {
        0: [0, 0],
        1: [0, 1],
        2: [0, 2],
        3: [1, 0],
        4: [1, 1],
        5: [1, 2],
        6: [2, 0],
        7: [2, 1],
        8: [2, 2],
        9: [3, 0],
        10: [3, 1],
        11: [3, 2],
        12: [4, 0],
        13: [4, 1],
        14: [4, 2],
        }

# compute basic statistics from the data
for name, inv in [
        ('PHOIBLE', stmD), 
        ('UPSID', upsD),
        ('JIPA', jpaD),
        ('LAPSYD', lpsD)]:
    count = len(inv)
    all_count = 0
    for i in inv.values():
        all_count += len(i)
    print(name, count, all_count)

# coverage for four datasets
coverage = [[0 for x in range(4)] for y in range(4)]

# plot the deltas
for (idx, nameA, dataA, dictA), (jdx, nameB, dataB, dictB) in progressbar(combinations(
        [
            (0, 'Phoible', stm, stmD), 
            (1, 'JIPA', jpa, jpaD), 
            (2, 'LAPSYD', lps, lpsD),
            (3, 'UPSID', ups, upsD)], 
        r=2)):
    matches = [k for k in dataA if k in dataB]

    for i, param in enumerate(['Sounds', 'Consonantal', 'Vocalic']):
        lstA, lstB, values = [], [], []

        for gcode in matches:
            vA, vB = dataA[gcode][param], dataB[gcode][param]
            if isinstance(vA, (int, float)) and isinstance(vB, (int, float)):
                lstA += [vA]
                lstB += [vB]
                values += [gcode]
        if values:
            fig = plt.Figure()
            plt.hist([x-y for x, y in zip(lstA, lstB)])
            plt.title(param)
            plt.savefig('plots/delta-{0}-{1}-{2}.pdf'.format(nameA, nameB,
                param))
            plt.clf()
            

for (idx, nameA, dataA, dictA), (jdx, nameB, dataB, dictB) in progressbar(combinations(
        [
            (0, 'Phoible', stm, stmD), 
            (1, 'JIPA', jpa, jpaD), 
            (2, 'LAPSYD', lps, lpsD),
            (3, 'UPSID', ups, upsD)], 
        r=2)):
    #fig, axs = plt.subplots(2, 3)
    table = []
    matches = [k for k in dataA if k in dataB]
    coverage[idx][idx] = len(dataA)
    coverage[jdx][jdx] = len(dataB)
    coverage[idx][jdx] = len(matches)
    coverage[jdx][idx] = len(matches) / min([len(dataA), len(dataB)])
    for i, param in enumerate(parameters):
        lstA, lstB, values = [], [], []
        for gcode in matches:
            vA, vB = dataA[gcode][param], dataB[gcode][param]
            if isinstance(vA, (int, float)) and isinstance(vB, (int, float)):
                lstA += [vA]
                lstB += [vB]
                values += [gcode]
        if values:
            p, r = spearmanr(lstA, lstB)
            d = deltas(lstA, lstB)
            if param in ['Sounds', 'Consonants', 'Vowels']:
                strict = compare_inventories(dictA, dictB, aspects=[param.lower()])
                approx = compare_inventories(dictA, dictB,
                        aspects=[param.lower()], similarity='approximate')
            elif param == 'Consonantal':
                strict = compare_inventories(dictA, dictB, aspects=['consonants', 'clusters'])
                approx = compare_inventories(dictA, dictB,
                        aspects=['consonants', 'clusters'],
                        similarity='approximate')
            elif param == 'Vocalic':
                strict = compare_inventories(dictA, dictB, aspects=['vowels',
                    'diphthongs'])
                approx = compare_inventories(dictA, dictB,
                        aspects=['vowels', 'diphthongs'],
                        similarity='approximate')
            else:
                strict = 0
                approx = 0
            plt.plot(lstA, lstB, '.', color='crimson')
            plt.title(param)
            plt.xlabel(nameA)
            plt.ylabel(nameB)
            plt.xlim(0, max(lstA+lstB)+5)
            plt.ylim(0, max(lstA+lstB)+5)
            plt.savefig('plots/{0}-{1}-{2}.pdf'.format(nameA, nameB, param))
            plt.clf()
            #this_ax = axs[idxs[i][0], idxs[i][1]]
            #this_ax.plot(lstA, lstB, '.')
            #this_ax.set(title=param)
            table += [[param, p, r, d, strict, approx, len(values)]]
            
                
    #for ax in axs.flat:
    #    ax.set(xlabel=nameA)
    #    ax.set(ylabel=nameB)
    #plt.savefig('plots/plots-{0}-{1}.pdf'.format(nameA, nameB))
    print('\n# {0} / {1}'.format(nameA, nameB))
    print(tabulate(
        table, floatfmt='.4f', 
        headers=['Correlation', 'P-Value', 'Deltas', 'StrictSim', 'ApproxSim', 'Sample']
        ))
print(tabulate(coverage))

