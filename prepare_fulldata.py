from pathlib import Path
import csv
from pyclts import CLTS
from pyclts.inventories import Inventory


def read_data():
    # load basic data
    clts = CLTS()
    bipa = clts.transcriptionsystem_dict["bipa"]

    # Get the path to the directory containing the source files.
    BASE_PATH = Path(__file__).parent
    sources = {
        filename.stem.split("-")[0].lower(): filename
        for filename in BASE_PATH.glob("*-data.tsv")
    }

    # Read the contents of all source files, add a column with source ("dataset") name,
    # and aggregate all rows into a single list of dictionaries.
    rows = []
    for dataset, filename in sources.items():
        with open(filename, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["Dataset"] = dataset
                # Convert "Latitude", "Longitude", and "Ratio" to floats if they exist, None otherwise
                for key in ["Latitude", "Longitude", "Ratio"]:
                    if row[key]:
                        row[key] = float(row[key])
                    else:
                        row[key] = None
                rows.append(row)

    # For each row, build an inventory object and add columns with the sorted lists of
    # consonants, vowels, and diphthongs. This will also replace the "Phonemes" column
    # with a sorted list of its phonemes.
    for row in rows:
        phonemes = sorted(row["Phonemes"].split())
        inv = Inventory.from_list(
            *phonemes, id=row["ID"], language=row["Glottocode"], ts=bipa
        )
        consonants = sorted(inv.consonants.keys())
        vowels = sorted(inv.vowels.keys())
        diphs = sorted(inv.diphthongs.keys())

        row["Num_Phonemes"] = row.pop("Sounds")
        row["Num_Consonants"] = len(consonants)
        row["Num_Clusters"] = row.pop("Clusters")
        row["Num_Vowels"] = len(vowels)
        row["Num_Diphthongs"] = len(diphs)
        row["Num_Long_Consonants"] = row.pop("LongConsonants")
        row["Num_Long_Vowels"] = row.pop("LongVowels")

        row["Phonemes"] = " ".join(phonemes)
        row["Consonants"] = " ".join(consonants)
        row["Vowels"] = " ".join(vowels)
        row["Diphthongs"] = " ".join(diphs)

    # Sort rows
    rows = sorted(
        rows, key=lambda row: (row["Glottocode"], row["Dataset"], row["Name"])
    )

    return rows


def main():
    # Get full data
    full_data = read_data()

    # Write the full data to a file, clipping all floating point numbers to 4 decimal places.
    tabular_keys = [
        "ID",
        "Glottocode",
        "Dataset",
        "Name",
        "Family",
        "Macroarea",
        "Latitude",
        "Longitude",
        "Num_Phonemes",
        "Num_Consonants",
        "Num_Vowels",
        "Num_Clusters",
        "Num_Long_Consonants",
        "Num_Long_Vowels",
        "Num_Diphthongs",
        "Ratio",
        "Phonemes",
        "Consonants",
        "Vowels",
        "Diphthongs",
    ]
    with open("fulldata.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=tabular_keys)
        writer.writeheader()
        for row in full_data:
            writer.writerow(
                {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()}
            )


if __name__ == "__main__":
    main()
