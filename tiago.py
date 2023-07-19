from collections import Counter
from itertools import combinations, product
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import pickle
import os.path
from scipy.stats import spearmanr
import copy

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


def collect_results_datasets(data):
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
            statistics[col + "_Proportion_NonZero"] = (group[col] > 0).mean()
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
        statistics_df[col + "_Mean"] = statistics_df[col + "_Mean"].map("{:.4f}".format)
        statistics_df[col + "_All_Proportion"] = statistics_df[
            col + "_All_Proportion"
        ].map("{:.4f}".format)
        statistics_df[col + "_Dataset_Proportion"] = statistics_df[
            col + "_Dataset_Proportion"
        ].map("{:.4f}".format)
        statistics_df[col + "_Proportion_NonZero"] = statistics_df[
            col + "_Proportion_NonZero"
        ].map("{:.4f}".format)
    statistics_df["Global_All_Proportion"] = statistics_df["Global_All_Proportion"].map(
        "{:.4f}".format
    )
    statistics_df["Global_Dataset_Proportion"] = statistics_df[
        "Global_Dataset_Proportion"
    ].map("{:.4f}".format)

    # Sort DataFrame
    statistics_df.sort_values(
        ["Dataset", "Macroarea"],
        key=lambda col: col == "GLOBAL" if isinstance(col, str) else col,
        inplace=True,
    )

    # Drop unnecessary columns
    statistics_df.drop(["Num_Phonemes_Proportion_NonZero"], axis=1, inplace=True)

    return statistics_df


def _get_distance_cache(data):
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

    return distance_cache


def collect_results_comparisons(data):
    # Drop all rows with "GLOBAL" macroarea
    data = data[data["Macroarea"] != "GLOBAL"]

    distance_cache = _get_distance_cache(data)
    datasets = sorted(data["Dataset"].unique())
    result = []

    # Setup caching dictionaries
    cache_js = {}
    cache_strict = {}
    cache_approx = {}

    # Setup inventory cache
    inventory_cache = {
        row["ID"]: get_clts_inventory(row["Phonemes_split"])
        for index, row in data.iterrows()
    }

    for dataset_a, dataset_b in tqdm(list(itertools.permutations(datasets, 2))):
        data_a = data[data["Dataset"] == dataset_a]
        data_b = data[data["Dataset"] == dataset_b]

        glottocodes_a = set(data_a["Glottocode"])
        glottocodes_b = set(data_b["Glottocode"])
        shared_glottocodes = glottocodes_a.intersection(glottocodes_b)

        num_comparable_glottocodes = len(shared_glottocodes)

        comparable_inventories_a = data_a[data_a["Glottocode"].isin(shared_glottocodes)]
        comparable_inventories_b = data_b[data_b["Glottocode"].isin(shared_glottocodes)]

        num_comparable_inventories_a = len(comparable_inventories_a)
        num_comparable_inventories_b = len(comparable_inventories_b)

        mean_global_a = round(np.mean(data_a["Num_Phonemes"]), 4)
        mean_global_b = round(np.mean(data_b["Num_Phonemes"]), 4)
        mean_comparable_a = round(np.mean(comparable_inventories_a["Num_Phonemes"]), 4)
        mean_comparable_b = round(np.mean(comparable_inventories_b["Num_Phonemes"]), 4)

        list_of_phonemes_a = [
            phoneme
            for inventory in comparable_inventories_a["Phonemes_split"]
            for phoneme in inventory
        ]
        list_of_phonemes_b = [
            phoneme
            for inventory in comparable_inventories_b["Phonemes_split"]
            for phoneme in inventory
        ]

        # Convert to frequency distribution and compute the Jensen-Shannon divergence
        counts_a = Counter(list_of_phonemes_a)
        counts_b = Counter(list_of_phonemes_b)
        all_phonemes = set(list_of_phonemes_a).union(set(list_of_phonemes_b))
        vec_a = np.array([counts_a[phoneme] for phoneme in all_phonemes])
        vec_b = np.array([counts_b[phoneme] for phoneme in all_phonemes])
        p_a = vec_a / np.sum(vec_a)
        p_b = vec_b / np.sum(vec_b)
        comparable_jensenshannon = round(jensenshannon(p_a, p_b), 4)

        dummy_inv_a = get_clts_inventory(list_of_phonemes_a)
        dummy_inv_b = get_clts_inventory(list_of_phonemes_b)

        comparable_strict = round(dummy_inv_a.strict_similarity(dummy_inv_b), 4)
        comparable_approx = round(
            dummy_inv_a.tiago_approximate_similarity(
                dummy_inv_b, distance_cache=distance_cache
            ),
            4,
        )

        # Convert the lists of phonemes into Counter objects
        counter_a = Counter(list_of_phonemes_a)
        counter_b = Counter(list_of_phonemes_b)

        # Identify the set of all unique phonemes in either of the lists
        all_phonemes = set(list_of_phonemes_a).union(set(list_of_phonemes_b))

        # Map the frequency of phonemes in each Counter object to the reference set
        vec_a = np.array([counter_a[phoneme] for phoneme in all_phonemes])
        vec_b = np.array([counter_b[phoneme] for phoneme in all_phonemes])

        # Compute the Spearman correlation for the counts of phonemes in the two lists
        r_counts, p_counts = spearmanr(vec_a, vec_b)
        r_counts = format(r_counts, ".4f")
        p_counts = format(p_counts, ".8f")

        aggregated_js = []
        aggregated_strict = []
        aggregated_approx = []
        aggregated_size_diff = []
        aggregated_median_size_a = []
        aggregated_median_size_b = []

        for glottocode in shared_glottocodes:
            # Obtain the mean inventory size for the current glottocode in both datasets
            median_size_a = np.median(
                comparable_inventories_a[
                    comparable_inventories_a["Glottocode"] == glottocode
                ]["Num_Phonemes"]
            )
            median_size_b = np.median(
                comparable_inventories_b[
                    comparable_inventories_b["Glottocode"] == glottocode
                ]["Num_Phonemes"]
            )
            aggregated_median_size_a.append(median_size_a)
            aggregated_median_size_b.append(median_size_b)

            # Get the subset of the data for the current glottocode
            subset_a = comparable_inventories_a[
                comparable_inventories_a["Glottocode"] == glottocode
            ]
            subset_b = comparable_inventories_b[
                comparable_inventories_b["Glottocode"] == glottocode
            ]

            # Get the phoneme lists for the current glottocode, and compute the
            # Jensen-Shannon divergence, the strict similarity, the approximate
            # similarity, and the size difference
            for row_a, row_b in product(subset_a.iterrows(), subset_b.iterrows()):
                phonemes_a = row_a[1]["Phonemes_split"]
                phonemes_b = row_b[1]["Phonemes_split"]

                # Get the ID frozenset for the cache
                id_frozenset = frozenset([row_a[1]["ID"], row_b[1]["ID"]])

                # Compute the size difference
                if id_frozenset in cache_js:
                    js_divergence = cache_js[id_frozenset]
                else:
                    # Convert to frequency distribution and compute the Jensen-Shannon divergence
                    counts_a = Counter(phonemes_a)
                    counts_b = Counter(phonemes_b)
                    all_phonemes = set(phonemes_a).union(set(phonemes_b))
                    vec_a = np.array([counts_a[phoneme] for phoneme in all_phonemes])
                    vec_b = np.array([counts_b[phoneme] for phoneme in all_phonemes])
                    p_a = vec_a / np.sum(vec_a)
                    p_b = vec_b / np.sum(vec_b)
                    js_divergence = round(jensenshannon(p_a, p_b), 4)
                    cache_js[id_frozenset] = js_divergence
                aggregated_js.append(js_divergence)

                # Compute the strict similarity
                if id_frozenset in cache_strict:
                    strict_sim = cache_strict[id_frozenset]
                else:
                    strict_sim = round(
                        inventory_cache[row_a[1]["ID"]].strict_similarity(
                            inventory_cache[row_b[1]["ID"]]
                        ),
                        4,
                    )
                    cache_strict[id_frozenset] = strict_sim
                aggregated_strict.append(strict_sim)

                # Compute the approximate similarity
                if id_frozenset in cache_approx:
                    approx_sim = cache_approx[id_frozenset]
                else:
                    approx_sim = round(
                        inventory_cache[row_a[1]["ID"]].tiago_approximate_similarity(
                            inventory_cache[row_b[1]["ID"]],
                            distance_cache=distance_cache,
                        ),
                        4,
                    )
                    cache_approx[id_frozenset] = approx_sim
                aggregated_approx.append(approx_sim)

                # Using the number of phonemes, compute the size difference
                size_diff = len(phonemes_a) - len(phonemes_b)
                aggregated_size_diff.append(size_diff)

        # Compute the spearman correlation for the median inventory sizes
        r_median, p_median = spearmanr(
            aggregated_median_size_a, aggregated_median_size_b
        )
        r_median = format(r_median, ".4f")
        p_median = format(p_median, ".8f")

        result.append(
            [
                dataset_a,
                dataset_b,
                num_comparable_glottocodes,
                num_comparable_inventories_a,
                num_comparable_inventories_b,
                mean_global_a,
                mean_global_b,
                format(mean_global_a - mean_global_b, ".4f"),
                mean_comparable_a,
                mean_comparable_b,
                format(mean_comparable_a - mean_comparable_b, ".4f"),
                format(mean_comparable_a / mean_global_a, ".4f"),
                format(mean_comparable_b / mean_global_b, ".4f"),
                comparable_jensenshannon,
                comparable_strict,
                comparable_approx,
                r_median,
                p_median,
                r_counts,
                p_counts,
                format(np.mean(aggregated_js), ".4f"),
                format(np.std(aggregated_js), ".4f"),
                format(np.mean(aggregated_strict), ".4f"),
                format(np.std(aggregated_strict), ".4f"),
                format(np.mean(aggregated_approx), ".4f"),
                format(np.std(aggregated_approx), ".4f"),
                format(np.mean(aggregated_size_diff), ".4f"),
                format(np.std(aggregated_size_diff), ".4f"),
            ]
        )

    df = pd.DataFrame(
        result,
        columns=[
            "Dataset_A",
            "Dataset_B",
            "Num_Comparable_Glottocodes",
            "Num_Comparable_Inventories_A",
            "Num_Comparable_Inventories_B",
            "Mean_Global_A",
            "Mean_Global_B",
            "Mean_Global_Diff",
            "Mean_Comparable_A",
            "Mean_Comparable_B",
            "Mean_Comparable_Diff",
            "Mean_Comparable_A/Global_A",
            "Mean_Comparable_B/Global_B",
            "Comparable_JensenShannon",
            "Comparable_Strict",
            "Comparable_Approximate",
            "Spearman_R_InvSize",
            "Spearman_p_InvSize",
            "Spearman_R_Counts",
            "Spearman_p_Counts",
            "Aggregated_JensenShannon_Mean",
            "Aggregated_JensenShannon_SD",
            "Aggregated_Strict_Mean",
            "Aggregated_Strict_SD",
            "Aggregated_Approx_Mean",
            "Aggregated_Approx_SD",
            "Aggregated_Size_Diff_Mean",
            "Aggregated_Size_Diff_SD",
        ],
    )
    return df


def collect_phoneme_frequency(df):
    # Filter out Macroarea "GLOBAL" from the entire dataset
    df = df[df["Macroarea"] != "GLOBAL"]

    # filter out "ALL" dataset for global statistics
    df_all = df[df["Dataset"] != "ALL"]

    phoneme_data = []
    for dataset in df["Dataset"].unique():
        print(dataset)
        df_dataset = df[df["Dataset"] == dataset]
        dataset_phoneme_counts = df_dataset["Phonemes_split"].explode().value_counts()

        for phoneme in tqdm(dataset_phoneme_counts.index):
            dataset_count = dataset_phoneme_counts[phoneme]
            global_count = (
                df_all["Phonemes_split"].explode().value_counts().get(phoneme, 0)
            )

            dataset_ratio = round(dataset_count / dataset_phoneme_counts.sum(), 4)
            global_ratio = round(
                global_count / df_all["Phonemes_split"].explode().value_counts().sum(),
                4,
            )

            dataset_inv_ratio = round(
                (df_dataset["Phonemes_split"].apply(lambda x: phoneme in x).mean()), 4
            )
            global_inv_ratio = round(
                (df_all["Phonemes_split"].apply(lambda x: phoneme in x).mean()), 4
            )

            phoneme_dataset_ratio = round(
                (
                    df_all[df_all["Phonemes_split"].apply(lambda x: phoneme in x)][
                        "Dataset"
                    ].nunique()
                    / df_all["Dataset"].nunique()
                ),
                4,
            )

            phoneme_data.append(
                [
                    phoneme,
                    dataset,
                    dataset_count,
                    global_count,
                    dataset_ratio,
                    global_ratio,
                    dataset_inv_ratio,
                    global_inv_ratio,
                    phoneme_dataset_ratio,
                ]
            )

    result = pd.DataFrame(
        phoneme_data,
        columns=[
            "Phoneme",
            "Dataset",
            "Dataset_Count",
            "Global_Count",
            "Dataset_Ratio",
            "Global_Ratio",
            "Dataset_InvRatio",
            "Global_InvRatio",
            "Phoneme_Dataset_Ratio",
        ],
    )

    # Sort the DataFrame
    result = result.sort_values(by=["Phoneme", "Dataset"]).reset_index(drop=True)

    return result


def main():
    # Get data
    data = get_data()

    # Get summary statistics
    # summary_df = build_summary(data)
    # summary_df.to_csv("tiago.summary.tsv", sep="\t", index=False)

    # Get frequency statistics
    # phoneme_stats_df = build_phoneme_stats(data)
    # phoneme_stats_df.to_csv("tiago.phoneme_stats.tsv", sep="\t", index=False)

    # Make a copy of the data and get the "results_datasets" dataframe
    data_copy = copy.deepcopy(data)
    results_df = collect_results_datasets(data_copy)
    results_df.to_csv("tiago.results_datasets.tsv", sep="\t", index=False)

    # Make a copy of the data and get the "results_comparisons" dataframe
    data_copy = copy.deepcopy(data)
    comparisons_df = collect_results_comparisons(data_copy)
    comparisons_df.to_csv("tiago.results_comparisons.tsv", sep="\t", index=False)

    # Make a copy of the data and get the "phoneme_frequency" dataframe
    data_copy = copy.deepcopy(data)
    phoneme_frequency_df = collect_phoneme_frequency(data_copy)
    phoneme_frequency_df.to_csv("tiago.phoneme_frequency.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
