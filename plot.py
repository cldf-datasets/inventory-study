from tabulate import tabulate
from statistics import median, mean
from itertools import combinations, product
from pyclts.inventories import Inventory
from pyclts import CLTS
from tqdm import tqdm as progressbar

from cartopy import *
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
from itertools import product
import cartopy.feature as cfeature
from collections import defaultdict
from csvw.dsv import UnicodeDictReader

from matplotlib import pyplot as plt
from matplotlib import cm


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
    coords = {}
    for row in data.values():
        if row["Latitude"]:
            gcodes[row["Glottocode"]] += [
                Inventory.from_list(
                    *row["Phonemes"].split(" "), language=row["ID"], ts=ts
                )
            ]
        coords[row["Glottocode"]] = row
    return gcodes, coords


def transform(val, threshold=20):
    if val < 0:
        return 0.5 - abs(val) / threshold
    elif val > 0:
        return 0.5 + val / threshold
    else:
        return 0.5


bipa = CLTS().bipa

parameters = ["Sounds", "Consonants", "Vowels", "Consonantal", "Vocalic", "Ratio"]

(
    (jipa_data, jipa_codes, jipa),
    (lapsyd_data, lapsyd_codes, lapsyd),
    (upsid_data, upsid_codes, upsid),
    (phoible_data, phoible_codes, phoible),
    (aa_data, aa_codes, aa),
    (ra_data, ra_codes, ra),
    (saphon_data, saphon_codes, saphon),
    (ea_data, ea_codes, ea),
    (er_data, er_codes, er),
) = [
    to_dict(ds, parameters)
    for ds in ["jipa", "lapsyd", "UPSID", "PHOIBLE", "AA", "RA", "SAPHON", "EA", "ER"]
]


(
    (jpaD, jpaC),
    (lpsD, lpsC),
    (upsD, upsC),
    (phoibleD, phoibleC),
    (aaD, aaC),
    (raD, raC),
    (saphonD, saphonC),
    (eaD, eaC),
    (erD, erC),
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


# plot the deltas
for (idx, nameA, dataA, dictA, coordsA), (
    jdx,
    nameB,
    dataB,
    dictB,
    coordsB,
) in progressbar(
    combinations(
        [
            (0, "JIPA", jipa, jpaD, jpaC),
            (1, "LAPSYD", lapsyd, lpsD, lpsC),
            (2, "UPSID", upsid, upsD, upsC),
            (3, "PHOIBLE", phoible, phoibleD, phoibleC),
            (4, "AA", aa, aaD, aaC),
            (5, "RA", ra, raD, raC),
            (6, "SAPHON", saphon, saphonD, saphonC),
            (7, "EA", ea, eaD, eaC),
            (8, "ER", er, erD, erC),
        ],
        r=2,
    )
):
    matches = [k for k in dataA if k in dataB]

    for i, param in enumerate(["Sounds"]):
        lstA, lstB, values = [], [], []

        for gcode in matches:
            vA, vB = dataA[gcode][param], dataB[gcode][param]
            if isinstance(vA, (int, float)) and isinstance(vB, (int, float)):
                lstA += [vA]
                lstB += [vB]
                values += [
                    (gcode, coordsA[gcode]["Latitude"], coordsB[gcode]["Longitude"])
                ]

        if values:
            fig = plt.Figure(figsize=[20, 10])
            ax = plt.axes(projection=ccrs.PlateCarree())
            # fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            # stamen_terrain = cimgt.Stamen('terrain-background')
            # ax.add_image(stamen_terrain, 5)
            # ax.set_extent([-25, 50, 30, 70], crs=ccrs.PlateCarree())
            # ax.coastlines(resolution='110m')
            # ax.stock_img()
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            # ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            ax.add_feature(cfeature.RIVERS)
            for vA, vB, (g, lat, lon) in zip(lstA, lstB, values):
                score = transform(vA - vB)
                print(score, vA, vB, vA - vB)
                color = cm.bwr(score)
                try:
                    plt.plot(
                        float(lon),
                        float(lat),
                        marker="o",
                        markersize=5,
                        color=color,
                        markeredgewidth=1,
                        markeredgecolor="black",
                    )
                except:
                    print(g)

            plt.title(
                "Comparing sound inventories for {0} vs. {1}".format(nameA, nameB)
            )
            plt.colorbar(
                cm.ScalarMappable(norm=None, cmap=cm.bwr),
                ax=ax,
                orientation="horizontal",
                shrink=0.4,
                # ticks=[-20, -10, 0, 10, 20]
            )
            plt.savefig("plots/map-{0}-{1}.pdf".format(nameA, nameB))
            plt.clf()
