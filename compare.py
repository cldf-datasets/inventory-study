from csvw import UnicodeDictReader, UnicodeWriter
from scipy.stats import spearmanr
from collections import defaultdict
from matplotlib import pyplot as plt
from tabulate import tabulate
from statistics import median, mean
from itertools import combinations, product
from pyclts.inventories import Inventory
from pyclts import CLTS
from tqdm import tqdm as progressbar
from sys import argv


def to_dict(path, parameters):
    with UnicodeDictReader(path + "-data.tsv", delimiter="\t") as reader:
        data = {row["ID"]: row for row in reader}
    gcodes = defaultdict(list)
    for row in data.values():
        gcodes[row["Glottocode"]] += [row["ID"]]

    # get the median per feature per glottocode
    values = {k: {p: "" for p in parameters} for k in gcodes}
    for gcode, varieties in gcodes.items():
        for p in parameters:
            param = []
            for variety in varieties:
                if data[variety].get(p, ""):
                    param += [float(data[variety][p])]
            if param:
                values[gcode][p] = median(param)
    return data, gcodes, values


def inventories(path, ts):
    with UnicodeDictReader(path + "-data.tsv", delimiter="\t") as reader:
        data = {row["ID"]: row for row in reader}
    gcodes = defaultdict(list)
    for row in data.values():
        gcodes[row["Glottocode"]] += [
            Inventory.from_list(*row["Phonemes"].split(" "), language=row["ID"], ts=ts)
        ]
    return gcodes


def deltas(lstA, lstB):
    score = 0
    for a, b in zip(lstA, lstB):
        score += abs(a - b)
    return score / len(lstA)


def compare_inventories(dctA, dctB, aspects, similarity="strict"):
    scores = []
    for code in dctB:
        if code in dctA:
            invsA, invsB = dctA[code], dctB[code]
            score = []
            for invA, invB in product(invsA, invsB):
                if similarity == "strict":
                    score += [invA.strict_similarity(invB, aspects=aspects)]
                else:
                    score += [invA.approximate_similarity(invB, aspects=aspects)]
            score = mean(score)
            scores += [score]
    return mean(scores)


bipa = CLTS("./clts").bipa

parameters = ["Sounds", "Consonants", "Vowels"]

(
    (jpa_data, jpa_codes, jpa),
    (lps_data, lps_codes, lps),
    (ups_data, ups_codes, ups),
    (uz_data, uz_codes, uz),
    (ph_data, ph_codes, ph),
    (gm_data, gm_codes, gm),
) = [to_dict(ds, parameters) for ds in ["jipa", "lapsyd", "UPSID", "UZ", "PH", "GM"]]


jpaD, lpsD, upsD, uzD, phD, gmD = (
    inventories("jipa", bipa),
    inventories("lapsyd", bipa),
    inventories("UPSID", bipa),
    inventories("UZ", bipa),
    inventories("PH", bipa),
    inventories("GM", bipa),
)

all_gcodes = defaultdict(list)
for ds, dct in [
    ("jipa", jpaD),
    ("lapsyd", lpsD),
    ("upsid", upsD),
    ("uz", uzD),
    ("ph", phD),
    ("gm", gmD),
]:
    for code, invs in dct.items():
        all_gcodes[code] += [(ds, inv) for inv in invs]
with open("output/comparable-inventories.tsv", "w") as f:
    f.write(
        "Glottocode\tLAPSyD\tLAPSyD_Var\tJIPA\tJIPA_Var\tUPSID\tUPSID_Var\tUZ\tUZ_Var\tPH\tPHVar\tGM\tGM_Var\n"
    )
    for code, invs in all_gcodes.items():
        if len(invs) > 1:
            f.write(code)
            dsets = [x[0] for x in invs]
            for ds in ["jipa", "lapsyd", "upsid", "uz", "ph", "gm"]:
                f.write(
                    "\t"
                    + str(dsets.count(ds))
                    + "\t"
                    + " ".join([inv.language for ds_, inv in invs if ds_ == ds])
                )
            f.write("\n")

with open("output/compared-inventories.tsv", "w") as f:
    f.write(
        "Glottocode\tDatasetA\tVarietyA\tSoundsA\tConsonantsA\tVowelsA\tDatasetB\tVarietyB\tSoundsB\tConsonantsB\tVowelsB\tStrictSimilarity\tAverageSimilarity\tInventoryA\tInventoryB\n"
    )
    for code, invs in [(x, y) for x, y in all_gcodes.items() if len(y) > 1]:
        for (dsA, invA), (dsB, invB) in combinations(invs, r=2):
            f.write(
                "\t".join(
                    [
                        code,
                        dsA,
                        invA.language,
                        str(len(invA.sounds)),
                        str(len(invA.consonant_sounds)),
                        str(len(invA.vowel_sounds)),
                        dsB,
                        invB.language,
                        str(len(invB.sounds)),
                        str(len(invB.consonant_sounds)),
                        str(len(invB.vowel_sounds)),
                        "{0:.2f}".format(
                            invA.strict_similarity(invB, aspects=["sounds"])
                        ),
                        "{0:.2f}".format(
                            invA.approximate_similarity(invB, aspects=["sounds"])
                        ),
                        " ".join(invA.sounds),
                        " ".join(invB.sounds),
                    ]
                )
                + "\n"
            )
print("[i] computed basic comparisons of all inventories")

# coverage for four datasets
coverage = [[0 for x in range(6)] for y in range(6)]

# store results for later
storage = {"raw": [], "summary": [], "table": defaultdict(dict)}
for (idx, nameA, dataA, dictA), (jdx, nameB, dataB, dictB) in progressbar(
    combinations(
        [
            (0, "JIPA", jpa, jpaD),
            (1, "LAPSyD", lps, lpsD),
            (2, "UPSID", ups, upsD),
            (3, "PH", ph, phD),
            (4, "UZ", uz, uzD),
            (5, "GM", gm, gmD),
        ],
        r=2,
    )
):
    # fig, axs = plt.subplots(2, 3)
    table = []
    matches = [k for k in dataA if k in dataB]
    coverage[idx][idx] = len(dataA)
    coverage[jdx][jdx] = len(dataB)
    coverage[idx][jdx] = len(matches)
    coverage[jdx][idx] = len(matches) / min([len(dataA), len(dataB)])
    compared = []

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
            if param in ["Sounds"]:
                strict = compare_inventories(dictA, dictB, aspects=[param.lower()])
                approx = compare_inventories(
                    dictA, dictB, aspects=[param.lower()], similarity="approximate"
                )
            elif param == "Consonants":
                strict = compare_inventories(dictA, dictB, aspects=["consonant_sounds"])
                approx = compare_inventories(
                    dictA,
                    dictB,
                    aspects=["consonant_sounds"],
                    similarity="approximate",
                )
            elif param == "Vowels":
                strict = compare_inventories(dictA, dictB, aspects=["vowel_sounds"])
                approx = compare_inventories(
                    dictA,
                    dictB,
                    aspects=["vowel_sounds"],
                    similarity="approximate",
                )
            else:
                strict = 0
                approx = 0

            # save data for later
            raw = [
                dict(zip(["SizeA", "SizeB", "Glottocode"], z))
                for z in zip(lstA, lstB, values)
            ]
            [
                raw[i].update({"Parameter": param, "DataA": nameA, "DataB": nameB})
                for i, r in enumerate(raw)
            ]
            storage["raw"].extend(raw)

            storage["summary"].append(
                {
                    "Parameter": param,
                    "Dataset1": nameA,
                    "Dataset2": nameB,
                    "P": p,
                    "R": r,
                    "Delta": d,
                    "Strict": strict,
                    "Approx": approx,
                }
            )

            table += [[param, p, r, d, strict, approx, len(values)]]

    print("\n# {0} / {1}".format(nameA, nameB))
    print(
        tabulate(
            table,
            floatfmt=".4f",
            headers=[
                "Correlation",
                "P-Value",
                "Deltas",
                "StrictSim",
                "ApproxSim",
                "Sample",
            ],
        )
    )

with UnicodeWriter("output/results.raw.csv") as writer:
    header = storage["raw"][0].keys()
    writer.writerow(header)
    for row in storage["raw"]:
        writer.writerow([row[h] for h in header])


with UnicodeWriter("output/results.summary.csv") as writer:
    header = storage["summary"][0].keys()
    writer.writerow(header)
    for row in storage["summary"]:
        writer.writerow([row[h] for h in header])
