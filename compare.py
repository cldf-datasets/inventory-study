from csvw import UnicodeDictReader, UnicodeWriter
from scipy.stats import spearmanr
from collections import defaultdict
from tabulate import tabulate
from statistics import median, mean
from itertools import combinations, product
from pyclts.inventories import Inventory
from pyclts import CLTS
from tqdm import tqdm as progressbar
import csv


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
    (jipa_data, jipa_codes, jipa_values),
    (lapsyd_data, lapsyd_codes, lapsyd_values),
    (upsid_data, upsid_codes, upsid_values),
    (phoible_data, phoible_codes, phoible_values),
    (aa_data, aa_codes, aa_values),
    (ra_data, ra_codes, ra_values),
    (saphon_data, saphon_codes, saphon_values),
    (ea_data, ea_codes, ea_values),
    (er_data, er_codes, er_values),
) = [
    to_dict(ds, parameters)
    for ds in [
        "jipa",
        "lapsyd",
        "UPSID",
        "PHOIBLE",
        "AA",
        "RA",
        "SAPHON",
        "EA",
        "ER",
    ]
]


(
    jipa_gcodes,
    lapsyd_gcodes,
    upsid_gcodes,
    phoible_gcodes,
    aa_gcodes,
    ra_gcodes,
    saphon_gcodes,
    ea_gcodes,
    er_gcodes,
) = (
    inventories("jipa", bipa),
    inventories("lapsyd", bipa),
    inventories("UPSID", bipa),
    inventories("PHOIBLE", bipa),
    inventories("AA", bipa),
    inventories("RA", bipa),
    inventories("SAPHON", bipa),
    inventories("EA", bipa),
    inventories("ER", bipa),
)

all_gcodes = defaultdict(list)
for ds, dct in [
    ("jipa", jipa_gcodes),
    ("lapsyd", lapsyd_gcodes),
    ("upsid", upsid_gcodes),
    ("phoible", phoible_gcodes),
    ("aa", aa_gcodes),
    ("ra", ra_gcodes),
    ("saphon", saphon_gcodes),
    ("ea", ea_gcodes),
    ("er", er_gcodes),
]:
    for code, invs in dct.items():
        all_gcodes[code] += [(ds, inv) for inv in invs]
with open("output/comparable-inventories.tsv", "w") as f:
    f.write(
        "Glottocode\tJIPA\tJIPA_Var\tLAPSyD\tLAPSyD_Var\tUPSID\tUPSID_Var\tPHOIBLE\tPHOIBLE_Var\tAA\tAA_Var\tRA\tRA_Var\tSAPHON\tSAPHON_Var\tEA\tEA_Var\tER\tER_Var\n"
    )
    for code, invs in all_gcodes.items():
        if len(invs) > 1:
            f.write(code)
            dsets = [x[0] for x in invs]
            for ds in [
                "jipa",
                "lapsyd",
                "upsid",
                "phoible",
                "aa",
                "ra",
                "saphon",
                "ea",
                "er",
            ]:
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

# coverage for the datasets
coverage = [[0 for x in range(9)] for y in range(9)]

# store results for later
storage = {"raw": [], "summary": [], "table": defaultdict(dict)}
for (idx, nameA, dataA, dictA), (jdx, nameB, dataB, dictB) in progressbar(
    combinations(
        [
            (0, "JIPA", jipa_values, jipa_gcodes),
            (1, "LAPSyD", lapsyd_values, lapsyd_gcodes),
            (2, "UPSID", upsid_values, upsid_gcodes),
            (3, "PHOIBLE", phoible_values, phoible_gcodes),
            (4, "AA", aa_values, aa_gcodes),
            (5, "RA", ra_values, ra_gcodes),
            (6, "SAPHON", saphon_values, saphon_gcodes),
            (7, "EA", ea_values, ea_gcodes),
            (8, "ER", er_values, er_gcodes),
        ],
        r=2,
    )
):
    # Print debugging information
    print("\n**************************************")
    print("[i] comparing", nameA, "and", nameB)
    print("[i] number of inventories in", nameA, ":", len(dataA))
    print("[i] number of inventories in", nameB, ":", len(dataB))
    print("[i] number of inventories in both:", len([k for k in dataA if k in dataB]))
    print(
        "[i] number of inventories in both (strict):",
        len([k for k in dataA if k in dataB and dataA[k] == dataB[k]]),
    )

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

# Names of all datasets
datasets = [
    "JIPA",
    "LAPSyD",
    "UPSID",
    "PHOIBLE",
    "AA",
    "RA",
    "SAPHON",
    "EA",
    "ER",
]

# Open CSV file
with open("output/mutual_coverage.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + datasets)  # Write header

    # Write rows
    for dataset, row in zip(datasets, coverage):
        writer.writerow([dataset] + row)
