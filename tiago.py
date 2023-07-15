import pandas as pd
from itertools import combinations, product


def build_summary(data):
    # Get unique dataset names
    dataset_names = data["Dataset"].unique()

    # Initialize list to hold the results
    results = []

    # Iterate over all combinations of datasets
    for dataset_a, dataset_b in combinations(dataset_names, 2):
        # Filter data for each dataset
        data_a = data[data["Dataset"] == dataset_a]
        data_b = data[data["Dataset"] == dataset_b]

        # Get glottocodes common to both datasets
        common_glottocodes = pd.Series(
            list(set(data_a["Glottocode"]).intersection(set(data_b["Glottocode"])))
        )
        num_common_glottocodes = len(common_glottocodes)

        # Get the number of inventories in each dataset with a common glottocode
        num_inventories_a = len(data_a[data_a["Glottocode"].isin(common_glottocodes)])
        num_inventories_b = len(data_b[data_b["Glottocode"].isin(common_glottocodes)])

        # Calculate mean size of the inventories in each dataset
        mean_size_a = data_a[data_a["Glottocode"].isin(common_glottocodes)][
            "Num_Phonemes"
        ].mean()
        mean_size_b = data_b[data_b["Glottocode"].isin(common_glottocodes)][
            "Num_Phonemes"
        ].mean()

        # Compute average intersection ratio for each common glottocode
        avg_intersection_ratios = []
        for glottocode in common_glottocodes:
            inventory_pairs = list(
                product(
                    data_a[data_a["Glottocode"] == glottocode]["Phonemes_split"],
                    data_b[data_b["Glottocode"] == glottocode]["Phonemes_split"],
                )
            )
            intersection_ratios = [
                len(set(a).intersection(b)) / len(set(a).union(b))
                for a, b in inventory_pairs
            ]
            avg_intersection_ratios.append(
                sum(intersection_ratios) / len(intersection_ratios)
            )

        avg_intersection_ratio = (
            sum(avg_intersection_ratios) / len(avg_intersection_ratios)
            if avg_intersection_ratios
            else None
        )

        # Append results
        results.append(
            [
                dataset_a,
                dataset_b,
                num_common_glottocodes,
                num_inventories_a,
                num_inventories_b,
                round(mean_size_a, 4) if mean_size_a is not None else None,
                round(mean_size_b, 4) if mean_size_b is not None else None,
                round(avg_intersection_ratio, 4)
                if avg_intersection_ratio is not None
                else None,
            ]
        )

    # Create a DataFrame from the results
    results_df = pd.DataFrame(
        results,
        columns=[
            "Dataset_A",
            "Dataset_B",
            "Glottocodes",
            "Inventories A",
            "Inventories B",
            "Mean size A",
            "Mean size B",
            "Average_Intersection_Ratio",
        ],
    )

    # Sort the DataFrame
    results_df = results_df.sort_values(by=["Dataset_A", "Dataset_B"]).reset_index(
        drop=True
    )

    return results_df


def build_phoneme_stats(data):
    # Flatten the lists of phonemes and calculate global statistics
    all_phonemes = pd.Series(
        [phoneme for sublist in data["Phonemes_split"] for phoneme in sublist]
    )
    global_counts = all_phonemes.value_counts()
    global_occurrences = global_counts
    global_total = global_counts.sum()
    global_ratio = global_counts / global_total
    global_inventory_counts = pd.Series(
        [
            phoneme
            for sublist in data["Phonemes_split"].apply(set).tolist()
            for phoneme in sublist
        ]
    ).value_counts()
    global_inventory_ratio = global_inventory_counts / len(data)

    # Initialize a list to store the results
    results = []

    # Iterate over all datasets
    for dataset in data["Dataset"].unique():
        data_subset = data[data["Dataset"] == dataset]

        # Flatten the lists of phonemes and calculate dataset-specific statistics
        dataset_phonemes = pd.Series(
            [
                phoneme
                for sublist in data_subset["Phonemes_split"]
                for phoneme in sublist
            ]
        )
        dataset_counts = dataset_phonemes.value_counts()
        dataset_occurrences = dataset_counts
        dataset_total = dataset_counts.sum()
        dataset_ratio = dataset_counts / dataset_total
        dataset_inventory_counts = pd.Series(
            [
                phoneme
                for sublist in data_subset["Phonemes_split"].apply(set).tolist()
                for phoneme in sublist
            ]
        ).value_counts()
        dataset_inventory_ratio = dataset_inventory_counts / len(data_subset)

        # Iterate over all phonemes
        for phoneme in all_phonemes.unique():
            # Get global and dataset-specific statistics
            results.append(
                [
                    phoneme,
                    dataset,
                    dataset_occurrences.get(phoneme, 0),
                    round(dataset_inventory_ratio.get(phoneme, 0), 4),
                    round(dataset_ratio.get(phoneme, 0), 4),
                    global_occurrences.get(phoneme),
                    round(global_inventory_ratio.get(phoneme), 4),
                    round(global_ratio.get(phoneme), 4),
                ]
            )

    # Create a DataFrame from the results
    stats = pd.DataFrame(
        results,
        columns=[
            "Phoneme",
            "Dataset",
            "Dataset_Occurrences",
            "Dataset_Inventory_Ratio",
            "Dataset_Ratio",
            "Global_Occurrences",
            "Global_Inventory_Ratio",
            "Global_Ratio",
        ],
    )

    # Sort the DataFrame
    stats = stats.sort_values(by=["Phoneme", "Dataset"]).reset_index(drop=True)

    return stats


def main():
    # Read the TSV file
    data = pd.read_csv("fulldata.tsv", delimiter="\t")

    # Split the "Phonemes" field into individual phonemes
    data["Phonemes_split"] = data["Phonemes"].str.split()

    # Get summary statistics
    summary_df = build_summary(data)
    summary_df.to_csv("tiago.summary.tsv", sep="\t", index=False)

    # Get frequency statistics
    phoneme_stats_df = build_phoneme_stats(data)
    phoneme_stats_df.to_csv("tiago.phoneme_stats.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
