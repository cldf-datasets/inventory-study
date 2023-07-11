from collections import Counter
from collections import defaultdict
from csvw import UnicodeDictReader, UnicodeWriter
from itertools import combinations, product
from pathlib import Path
from pyclts import CLTS
from pyclts.inventories import Inventory
from scipy.stats import spearmanr
from statistics import median, mean
from tabulate import tabulate
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

# Collect frequency stats for the graphemes used in each dataset
BASE_PATH = Path(__file__).parent
dataset_graphemes = {}
for dataset in BASE_PATH.glob("*-data.tsv"):
    # Extract the name of the dataset (e.g., "ER" from "ER-data.tsv")
    dataset_name = dataset.stem.split("-")[0]

    # Read the dataset and collect all grapheme occurrences
    with open(dataset, "r", encoding="utf-8") as f:
        tsv_reader = csv.DictReader(f, delimiter="\t")
        graphemes = Counter()
        for row in tsv_reader:
            graphemes.update(row["Phonemes"].split())

    # Store the counter
    dataset_graphemes[dataset_name] = graphemes

# Using the counters in `dataset_graphemes`, output a table with the
# absolute and relative frequencies of each grapheme in each dataset
with open(
    BASE_PATH / "output" / "grapheme_frequencies.tsv", "w", encoding="utf-8"
) as f:
    f.write("Grapheme\tDataset\tAbsolute\tRelative\n")
    for dataset, graphemes in dataset_graphemes.items():
        total = sum(graphemes.values())
        for grapheme, count in graphemes.most_common():
            f.write(f"{grapheme}\t{dataset}\t{count}\t{count / total:.4f}\n")

###############


def process_data(file_path):
    phoneme_counts = defaultdict(lambda: defaultdict(Counter))
    dataset_pair_counts = defaultdict(int)

    with open(file_path, "r") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        for row in reader:
            dataset_pair_AB = (row["DatasetA"], row["DatasetB"])
            dataset_pair_BA = (row["DatasetB"], row["DatasetA"])

            inventory_A = row["InventoryA"].split()
            inventory_B = row["InventoryB"].split()

            phoneme_counts[dataset_pair_AB]["DatasetA"].update(inventory_A)
            phoneme_counts[dataset_pair_BA]["DatasetB"].update(inventory_B)

            dataset_pair_counts[dataset_pair_AB] += 1
            dataset_pair_counts[dataset_pair_BA] += 1

    # Build results
    entries = []
    for pair, pair_counts in phoneme_counts.items():
        if pair[0] == pair[1]:
            continue

        total_A = sum(pair_counts["DatasetA"].values())
        total_B = sum(pair_counts["DatasetB"].values())

        for phoneme in set(
            list(pair_counts["DatasetA"].keys()) + list(pair_counts["DatasetB"].keys())
        ):
            count_A = pair_counts["DatasetA"][phoneme]
            count_B = pair_counts["DatasetB"][phoneme]

            rel_a = count_A / total_A if total_A > 0 else 0
            rel_b = count_B / total_B if total_B > 0 else 0

            percent_A = count_A / dataset_pair_counts[pair]
            percent_b = count_B / dataset_pair_counts[pair]

            entries.append(
                {
                    "Phoneme": phoneme,
                    "Dataset": pair[0],
                    "Dataset Pair": "-".join(pair),
                    "Absolute Count (DatasetA)": count_A,
                    "Absolute Count (DatasetB)": count_B,
                    "Relative Count (DatasetA)": "%.4f" % rel_a,
                    "Relative Count (DatasetB)": "%.4f" % rel_b,
                    "Total Inventories": dataset_pair_counts[pair],
                    "Percent (DatasetA)": "%.4f" % percent_A,
                    "Percent (DatasetB)": "%.4f" % percent_b,
                }
            )

    # Sort entries
    entries = sorted(
        entries,
        key=lambda x: (
            x["Phoneme"],
            x["Dataset"],
            x["Dataset Pair"],
            x["Absolute Count (DatasetA)"],
        ),
    )

    # Output results
    with open(
        BASE_PATH / "output" / "grapheme_frequencies.pairs.tsv",
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        fieldnames = [
            "Phoneme",
            "Dataset",
            "Dataset Pair",
            "Absolute Count (DatasetA)",
            "Absolute Count (DatasetB)",
            "Relative Count (DatasetA)",
            "Relative Count (DatasetB)",
            "Total Inventories",
            "Percent (DatasetA)",
            "Percent (DatasetB)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(entries)


# Call the function with your file path
process_data(BASE_PATH / "output" / "compared-inventories.tsv")

###############


def classify_grapheme(grapheme):
    # Convert the grapheme to a phoneme
    phoneme = bipa[grapheme]

    # Classify the phoneme based on its sound class and length
    if phoneme.type == "vowel":
        if phoneme.duration == "long":
            return 1  # 'long vowel'
        elif phoneme.duration is None:
            return 0  # 'normal vowel'
        else:
            return 2  # 'non-normal vowel with a length that is not long'
    elif phoneme.type == "consonant":
        if phoneme.duration == "long":
            return 5  # 'long consonant'
        elif phoneme.duration is None:
            return 4  # 'normal consonant'
        else:
            return 6  # 'non-normal consonant with a length that is not long'
    elif phoneme.type == "diphthong":
        return 3  #'diphthong'
    else:
        return 999  # 'unknown'


with open(BASE_PATH / "output" / "compared-inventories.tsv") as csvfile:
    reader = csv.DictReader(csvfile, delimiter="\t")
    rows = list(reader)

new_rows = []
for row in rows:
    inv_a = row["InventoryA"].split()
    inv_b = row["InventoryB"].split()

    # Sort the items in inv_a and inv_b by the value returned by classify_grapheme
    # first, and then by the grapheme itself
    inv_a.sort(key=lambda x: (classify_grapheme(x), x))
    inv_b.sort(key=lambda x: (classify_grapheme(x), x))

    # Join the sorted lists back into strings
    row["InventoryA"] = " ".join(inv_a)
    row["InventoryB"] = " ".join(inv_b)

    # Run `classify_grapheme` again, this time collecting graphemes by their
    # classification
    by_cat_a = defaultdict(list)
    by_cat_b = defaultdict(list)
    for grapheme in inv_a:
        by_cat_a[classify_grapheme(grapheme)].append(grapheme)
    for grapheme in inv_b:
        by_cat_b[classify_grapheme(grapheme)].append(grapheme)

    # Build a string for each classification, using the mapping in `cat_map`
    # to convert the classification number to a string
    cat_map = {
        0: "normal vowels",
        1: "long vowels",
        2: "odd-length vowels",
        3: "diphthongs",
        4: "normal consonants",
        5: "long consonants",
        6: "odd-length consonants",
        999: "unknowns",
    }
    for cat_idx in cat_map.keys():
        if cat_idx not in by_cat_a:
            row["%s A" % cat_map[cat_idx]] = ""
        else:
            row["%s A" % cat_map[cat_idx]] = " ".join(by_cat_a[cat_idx])
        if cat_idx not in by_cat_b:
            row["%s B" % cat_map[cat_idx]] = ""
        else:
            row["%s B" % cat_map[cat_idx]] = " ".join(by_cat_b[cat_idx])

    new_rows.append(row)

# Output the updated rows to a new file
with open(BASE_PATH / "output" / "compared-inventories.sorted.tsv", "w") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=list(new_rows[0].keys()),
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)
