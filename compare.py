from csvw.dsv import UnicodeDictReader
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from matplotlib import pyplot as plt
from tabulate import tabulate
from statistics import median
from itertools import combinations


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


parameters = ['Sounds', 'Consonants', 'Vowels', 'Consonantal', 'Vocalic', 'Ratio']

(
        (jpa_data, jpa_codes, jpa),
        (lps_data, lps_codes, lps),
        (ups_data, ups_codes, ups),
        (stm_data, stm_codes, stm)
        ) = [to_dict(ds, parameters) for ds in [
            'jipa', 'lapsyd', 'UPSID', 'UZ-PH-GM']]

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

for (nameA, dataA), (nameB, dataB) in combinations(
        [('Phoible', stm), ('JIPA', jpa), ('LAPSYD', lps), ('UPSID', ups)], r=2):

    fig, axs = plt.subplots(2, 3)
    table = []
    matches = [k for k in dataA if k in dataB]
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
            this_ax = axs[idxs[i][0], idxs[i][1]]
            this_ax.plot(lstA, lstB, '.')
            this_ax.set(title=param)
            table += [[param, p, r, len(values)]]
    for ax in axs.flat:
        ax.set(xlabel=nameA)
        ax.set(ylabel=nameB)
    plt.savefig('plots/plots-{0}-{1}.pdf'.format(nameA, nameB))
    print('\n# {0} / {1}'.format(nameA, nameB))
    print(tabulate(
        table, floatfmt='.4f', 
        headers=['Correlation', 'P-Value', 'Sample']
        ))

