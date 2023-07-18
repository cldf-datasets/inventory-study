from collections import defaultdict
from itertools import combinations, product
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import pickle
import os.path
from scipy.stats import spearmanr

from pyclts import CLTS
from pyclts.inventories import Inventory, Phoneme

BIPA = CLTS("./clts").bipa


def get_clts_inventory(phoneme_list):
    inv = Inventory.from_list(*phoneme_list, ts=BIPA)
    return inv


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


def get_data():
    # Read the TSV file
    data = pd.read_csv("fulldata.tsv", delimiter="\t")

    # Split the "Phonemes" field into individual phonemes
    data["Phonemes_split"] = data["Phonemes"].str.split()

    # Copy data and add a "GLOBAL" macroarea
    data_global = data.copy()
    data_global["Macroarea"] = "GLOBAL"
    data_extended_macroarea = pd.concat([data, data_global])

    # Copy data and add an 'ALL' dataset
    data_all = data_extended_macroarea.copy()
    data_all["Dataset"] = "ALL"
    data_extended = pd.concat([data_extended_macroarea, data_all])

    return data_extended


def collect_results(data):
    # List of the columns that we want to compute statistics for
    column_names = [
        "Num_Phonemes",
        "Num_Consonants",
        "Num_Vowels",
        "Num_Long_Consonants",
        "Num_Long_Vowels",
        "Num_Diphthongs",
    ]

    # Function to compute statistics
    def compute_statistics(group):
        statistics = pd.Series(dtype="float64")
        for col in column_names:
            statistics[col + "_Number"] = group[col].sum()
            statistics[col + "_Mean"] = group[col].mean()
        return statistics

    # Compute statistics
    statistics_df = (
        data.groupby(["Dataset", "Macroarea"]).apply(compute_statistics).reset_index()
    )

    # Compute proportions for each dataset in relation to "ALL"
    global_statistics = statistics_df[statistics_df["Dataset"] == "ALL"]
    for _, row in statistics_df.iterrows():
        dataset = row["Dataset"]
        area = row["Macroarea"]
        for col in column_names:
            statistics_df.loc[
                (statistics_df["Dataset"] == dataset)
                & (statistics_df["Macroarea"] == area),
                col + "_All_Proportion",
            ] = (
                row[col + "_Mean"]
                / global_statistics.loc[
                    global_statistics["Macroarea"] == area, col + "_Mean"
                ].values[0]
            )

    # Compute proportions for each dataset in relation to the global values of the dataset
    dataset_global_statistics = statistics_df[statistics_df["Macroarea"] == "GLOBAL"]
    for _, row in statistics_df.iterrows():
        dataset = row["Dataset"]
        area = row["Macroarea"]
        for col in column_names:
            dataset_global_statistic = dataset_global_statistics.loc[
                dataset_global_statistics["Dataset"] == dataset, col + "_Mean"
            ].values[0]
            if dataset_global_statistic:
                statistics_df.loc[
                    (statistics_df["Dataset"] == dataset)
                    & (statistics_df["Macroarea"] == area),
                    col + "_Dataset_Proportion",
                ] = (
                    row[col + "_Mean"] / dataset_global_statistic
                )

    # Compute Number_Inventories and Global_Proportion
    inventories_df = (
        data.groupby(["Dataset", "Macroarea"])
        .size()
        .reset_index(name="Number_Inventories")
    )
    statistics_df = pd.merge(statistics_df, inventories_df, on=["Dataset", "Macroarea"])
    total_inventories = statistics_df.loc[
        (statistics_df["Dataset"] == "ALL") & (statistics_df["Macroarea"] == "GLOBAL"),
        "Number_Inventories",
    ].values[0]
    statistics_df["Global_All_Proportion"] = (
        statistics_df["Number_Inventories"] / total_inventories
    )

    # Compute dataset proportion in relation to the global values of the dataset
    dataset_global_inventories = statistics_df[statistics_df["Macroarea"] == "GLOBAL"]
    for _, row in statistics_df.iterrows():
        dataset = row["Dataset"]
        area = row["Macroarea"]
        dataset_global_inventory = dataset_global_inventories.loc[
            dataset_global_inventories["Dataset"] == dataset, "Number_Inventories"
        ].values[0]
        if dataset_global_inventory:
            statistics_df.loc[
                (statistics_df["Dataset"] == dataset)
                & (statistics_df["Macroarea"] == area),
                "Global_Dataset_Proportion",
            ] = (
                row["Number_Inventories"] / dataset_global_inventory
            )

    # Convert Number_* fields to integer and round float fields to 4 decimal places
    for col in column_names:
        statistics_df[col + "_Number"] = statistics_df[col + "_Number"].astype(int)
        statistics_df[col + "_Mean"] = statistics_df[col + "_Mean"].round(4)
        statistics_df[col + "_All_Proportion"] = statistics_df[
            col + "_All_Proportion"
        ].round(4)
        statistics_df[col + "_Dataset_Proportion"] = statistics_df[
            col + "_Dataset_Proportion"
        ].round(4)
    statistics_df["Global_All_Proportion"] = statistics_df[
        "Global_All_Proportion"
    ].round(4)
    statistics_df["Global_Dataset_Proportion"] = statistics_df[
        "Global_Dataset_Proportion"
    ].round(4)

    # Sort DataFrame
    statistics_df.sort_values(
        ["Dataset", "Macroarea"],
        key=lambda col: col == "GLOBAL" if isinstance(col, str) else col,
        inplace=True,
    )

    return statistics_df


def collect_comparisons(data):
    datasets = data["Dataset"].unique()
    dataset_pairs = list(itertools.combinations(datasets, 2))

    # Initialize the distance and similarity caches
    if os.path.exists("sounds.distances"):
        with open("sounds.distances", "rb") as f:
            distance_cache = pickle.load(f)
    else:
        all_phonemes = set(
            phoneme for phonemes in data["Phonemes_split"] for phoneme in phonemes
        )
        distance_cache = {}
        for phoneme_a, phoneme_b in tqdm(itertools.combinations(all_phonemes, 2)):
            sound_a = BIPA[phoneme_a]
            sound_b = BIPA[phoneme_b]
            distance_cache[(phoneme_a, phoneme_b)] = sound_a.similarity(sound_b)
        with open("sounds.distances", "wb") as f:
            pickle.dump(distance_cache, f)

    js_divergence_cache = {}

    mean_global = data.groupby("Dataset")["Num_Phonemes"].mean()
    std_global = data.groupby("Dataset")["Num_Phonemes"].std()
    phoneme_counts = defaultdict(lambda: defaultdict(int))
    phoneme_sets = defaultdict(set)
    clts_inventory_cache = {}

    for idx, row in data.iterrows():
        for phoneme in row["Phonemes_split"]:
            phoneme_counts[row["Dataset"]][phoneme] += 1
            phoneme_sets[row["Dataset"]].add(phoneme)
        clts_inventory_cache[row["ID"]] = get_clts_inventory(row["Phonemes_split"])

    results = []
    for dataset_a, dataset_b in dataset_pairs:
        print(f"Comparing {dataset_a} and {dataset_b}")

        shared_glottocodes = set(
            data[data["Dataset"] == dataset_a]["Glottocode"]
        ).intersection(set(data[data["Dataset"] == dataset_b]["Glottocode"]))

        data_a = data[
            (data["Dataset"] == dataset_a)
            & (data["Glottocode"].isin(shared_glottocodes))
        ]
        data_b = data[
            (data["Dataset"] == dataset_b)
            & (data["Glottocode"].isin(shared_glottocodes))
        ]

        mean_compared_a = data_a["Num_Phonemes"].mean()
        std_compared_a = data_a["Num_Phonemes"].std()
        mean_compared_b = data_b["Num_Phonemes"].mean()
        std_compared_b = data_b["Num_Phonemes"].std()

        js_key = frozenset([dataset_a, dataset_b])
        if js_key in js_divergence_cache:
            js_divergence = js_divergence_cache[js_key]
        else:
            if len(data_a) > 0 and len(data_b) > 0:
                all_phonemes = sorted(
                    phoneme_sets[dataset_a].union(phoneme_sets[dataset_b])
                )
                counts_a = np.array(
                    [
                        phoneme_counts[dataset_a].get(phoneme, 0)
                        for phoneme in all_phonemes
                    ]
                )
                counts_b = np.array(
                    [
                        phoneme_counts[dataset_b].get(phoneme, 0)
                        for phoneme in all_phonemes
                    ]
                )
                freq_a = counts_a / np.sum(counts_a)
                freq_b = counts_b / np.sum(counts_b)
                js_divergence = jensenshannon(freq_a, freq_b)
                js_divergence_cache[js_key] = js_divergence
            else:
                js_divergence = np.nan

        strict_similarities = []
        approx_similarities = []
        combined_phonemes_a = []
        combined_phonemes_b = []
        for id_a, id_b in tqdm(itertools.product(data_a["ID"], data_b["ID"])):
            inv_a = clts_inventory_cache[id_a]
            inv_b = clts_inventory_cache[id_b]

            strict_sim = inv_a.strict_similarity(inv_b)
            approx_sim = inv_a.tiago_approximate_similarity(
                inv_b, distance_cache=distance_cache
            )

            strict_similarities.append(strict_sim)
            approx_similarities.append(approx_sim)

            combined_phonemes_a.extend(list(inv_a.sounds.keys()))
            combined_phonemes_b.extend(list(inv_b.sounds.keys()))

        # Compute strict and approximate similarity for combined inventories
        combined_inv_a = get_clts_inventory(combined_phonemes_a)
        combined_inv_b = get_clts_inventory(combined_phonemes_b)
        combined_strict_sim = combined_inv_a.strict_similarity(combined_inv_b)
        combined_approx_sim = combined_inv_a.tiago_approximate_similarity(
            combined_inv_b, distance_cache=distance_cache
        )

        # Average phoneme counts for each glottocode in each dataset
        data_a_phonemes = data_a.groupby("Glottocode")["Num_Phonemes"].mean().values
        data_b_phonemes = data_b.groupby("Glottocode")["Num_Phonemes"].mean().values

        if len(data_a_phonemes) > 0 and len(data_b_phonemes) > 0:
            r, p = spearmanr(data_a_phonemes, data_b_phonemes)
        else:
            r, p = np.nan, np.nan

        result = {
            "Dataset_A": dataset_a,
            "Dataset_B": dataset_b,
            "Comparable_Glottocodes": len(shared_glottocodes),
            "Comparable_Inventories_A": len(data_a),
            "Comparable_Inventories_B": len(data_b),
            "Comparable_Inventories_Total": len(data_a) + len(data_b),
            "Mean_Global_A": round(mean_global[dataset_a], 4),
            "Std_Global_A": round(std_global[dataset_a], 4),
            "Mean_Global_B": round(mean_global[dataset_b], 4),
            "Std_Global_B": round(std_global[dataset_b], 4),
            "Mean_Compared_A": round(mean_compared_a, 4),
            "Std_Compared_A": round(std_compared_a, 4),
            "Mean_Compared_B": round(mean_compared_b, 4),
            "Std_Compared_B": round(std_compared_b, 4),
            "Mean_Global_Diff": round(
                mean_global[dataset_a] - mean_global[dataset_b], 4
            ),
            "Mean_Compared_Diff": round(mean_compared_a - mean_compared_b, 4),
            "Jensen-Shannon": round(js_divergence, 4),
            "Mean_Strict_Similarity": round(np.mean(strict_similarities), 4),
            "Std_Strict_Similarity": round(np.std(strict_similarities), 4),
            "Mean_Approximate_Similarity": round(np.mean(approx_similarities), 4),
            "Std_Approximate_Similarity": round(np.std(approx_similarities), 4),
            "Combined_Strict_Similarity": round(combined_strict_sim, 4),
            "Combined_Approximate_Similarity": round(combined_approx_sim, 4),
            "R": round(r, 4),
            "P": round(p, 4),
        }

        results.append(result)

    return pd.DataFrame(results)


def main():
    # Get data
    data = get_data()

    # Get summary statistics
    summary_df = build_summary(data)
    summary_df.to_csv("tiago.summary.tsv", sep="\t", index=False)

    # Get frequency statistics
    phoneme_stats_df = build_phoneme_stats(data)
    phoneme_stats_df.to_csv("tiago.phoneme_stats.tsv", sep="\t", index=False)

    # Get results
    results_df = collect_results(data)
    results_df.to_csv("tiago.results_datasets.tsv", sep="\t", index=False)

    # Get comparisons
    comparisons_df = collect_comparisons(data)
    comparisons_df.to_csv("tiago.results_comparisons.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
