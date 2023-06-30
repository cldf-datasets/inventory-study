"""
Prepare the data for the sound inventories experiment.
"""
from cldfbench import get_dataset
import pycldf
from pyclts import CLTS
from pyclts.inventories import Inventory, Phoneme
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm as progressbar
from pyclts.util import nfd
from pyclts.transcriptionsystem import is_valid_sound
import json
import pybtex
from pycldf.sources import Source
from pylatexenc.latex2text import LatexNodes2Text
import attr


@attr.s
class Language:
    """
    Class is part of pylexibank, but not yet finished, so we reuse it here.
    """

    id = attr.ib()
    name = attr.ib()
    glottolog_name = attr.ib(default=None, repr=False)
    glottocode = attr.ib(default=None, repr=False)
    macroarea = attr.ib(default=None, repr=False)
    latitude = attr.ib(default=None, repr=False)
    longitude = attr.ib(default=None, repr=False)
    family = attr.ib(default=None, repr=False)
    forms = attr.ib(default=None, repr=False)
    attributes = attr.ib(default=None, repr=False)
    dataset = attr.ib(default=None, repr=False)

    def __len__(self):
        return len(self.forms)


def long_vowels(inventory):
    long_vowels = 0
    for sound in inventory.vowels.values():
        if "long" in sound.name or "ultra-long" in sound.name:
            long_vowels += 1
    return long_vowels


def long_consonants(inventory):
    long_consonants = 0
    for sound in inventory.consonants.values():
        if "long" in sound.name:
            long_consonants += 1
    return long_consonants


def normalize(grapheme):
    for s, t in [("\u2019", "\u02bc")]:
        grapheme = grapheme.replace(s, t)
    return grapheme


def get_cldf_varieties(dataset):
    """
    Load a generic CLDF dataset.
    """
    bipa = CLTS().bipa
    dset_ = get_dataset(dataset)
    dset = dset_.cldf_reader()
    try:
        bib = {source.id: source for source in dset_.cldf_dir.read_bib()}
    except:
        bib = {}
    dset = get_dataset(dataset).cldf_reader()
    languages = {row["ID"]: row for row in dset.iter_rows("LanguageTable")}
    params = {row["Name"]: row for row in dset.iter_rows("ParameterTable")}
    varieties = defaultdict(list)
    sources = defaultdict(set)
    for row in progressbar(dset.iter_rows("ValueTable"), desc="load values"):
        lid = row["Language_ID"]
        source = row["Source"][0] if row["Source"] else ""
        varieties[lid] += [nfd(row["Value"])]
        sources[lid].add(source)
    return languages, params, varieties, sources, bib


def get_phoible_varieties(
    dataset,
    path=Path(__file__).parent / "phoible" / "cldf",
):
    """
    Load phoible data (currently not in generic CLDF).
    """
    bipa = CLTS().bipa
    phoible = pycldf.Dataset.from_metadata(
        path.joinpath("StructureDataset-metadata.json")
    )
    bib = pybtex.database.parse_string(
        open(path.joinpath("sources.bib").as_posix()).read(), bib_format="bibtex"
    )
    bib_ = [Source.from_entry(k, e) for k, e in bib.entries.items()]
    bib = {source.id: source for source in bib_}
    gcodes = {row["ID"]: row for row in phoible.iter_rows("LanguageTable")}
    params = {row["Name"]: row for row in phoible.iter_rows("ParameterTable")}
    contributions = {
        row["ID"]: row["Inventory_source_ID"]
        for row in phoible.iter_rows("inventories.csv")
    }
    languages = {}
    varieties = defaultdict(list)
    sources = defaultdict(set)
    for row in progressbar(phoible.iter_rows("ValueTable"), desc="load values"):
        accept = False
        if contributions[row["Inventory_ID"]] == dataset:
            accept = True
        elif dataset == "PHOIBLE":
            if contributions[row["Inventory_ID"]] in ["PH", "UZ", "GM"]:
                accept = True

        if accept:
            lid = row["Language_ID"] + "-" + row["Inventory_ID"]
            varieties[lid] += [nfd(row["Value"])]
            languages[lid] = gcodes[row["Language_ID"]]
            source = row["Source"][0] if row["Source"] else ""
            sources[lid].add(source)
    return languages, params, varieties, sources, bib


def style_source(sources, bib):
    source = sources.pop()
    if source in bib:
        tmp = {}
        for k, v in bib[source].items():
            tmp[k.lower()] = LatexNodes2Text().latex_to_text(v)
        return "{0} ({1}): {2} [{3}]".format(
            tmp.get("author", "?"), tmp.get("year", "?"), tmp.get("title", "?"), source
        )
    elif source:
        print("Missing source {0}".format(source))
        return source
    return ""


def load_dataset(dataset, td=None, clts=None, dump=defaultdict(list)):
    clts = clts or CLTS()

    if not td:
        td = dataset

    if dataset in ["UPSID", "AA", "RA", "SAPHON", "EA", "ER", "PHOIBLE"]:
        dset_td = clts.transcriptiondata_dict["phoible"]
        languages, params, varieties, sources, bib = get_phoible_varieties(dataset)
    else:
        dset_td = clts.transcriptiondata_dict[td]
        languages, params, varieties, sources, bib = get_cldf_varieties(dataset)

    for sound in list(dset_td.grapheme_map):
        dset_td.grapheme_map[normalize(sound)] = dset_td.grapheme_map[sound]

    inventories = {}
    missing_gcodes = 0
    count = 0
    soundsD = defaultdict(int)
    exc_file = open("output/excluded.md", "a")
    exc_file.write("## Dataset {0}\n\n".format(dataset))
    for var, vals in progressbar(varieties.items(), desc="identify inventories"):
        for sound in vals:
            if sound not in dset_td.grapheme_map:
                bsound = bipa[sound]
                if bsound.type != "unknownsound" and is_valid_sound(bsound, bipa):
                    dset_td.grapheme_map[sound] = bipa[sound].s
        gcode = languages[var]

        if len(vals) == len(
            [v for v in vals if dset_td.grapheme_map.get(v, "<NA>") != "<NA>"]
        ):
            lang = Language(
                var,
                gcode["Name"],
                glottocode=gcode["Glottocode"],
                latitude=gcode["Latitude"],
                longitude=gcode["Longitude"],
                family=gcode.get("Family_Name", gcode.get("Family")),
                macroarea=gcode["Macroarea"],
                attributes=gcode,
            )

            sounds = {}
            for v in vals:
                s = dset_td.grapheme_map[v]
                b = bipa[s]
                if b.type in ["vowel", "consonant", "diphthong", "cluster"]:
                    try:
                        sounds[str(b)].graphemes_in_source += [v]
                    except KeyError:
                        sounds[str(b)] = Phoneme(
                            grapheme=str(b), graphemes_in_source=[v], occs=0, sound=b
                        )
                    dump["bipa-" + s] = b.name

            if lang.glottocode:
                inv = Inventory(id=var, sounds=sounds, language=lang, ts=bipa)
                inventories[var] = inv
                dump[gcode["Glottocode"]] += [
                    {
                        "ID": var,
                        "Dataset": dataset,
                        "Name": gcode["Name"],
                        "Source": style_source(sources[var], bib),
                        "CLTS": {
                            sound.grapheme: "//".join(sound.graphemes_in_source)
                            for sound in sounds.values()
                        },
                        "Sounds": vals,
                    }
                ]
            else:
                missing_gcodes += 1

        else:
            for sound in vals:
                if (
                    sound not in dset_td.grapheme_map
                    or dset_td.grapheme_map[sound] == "<NA>"
                ):
                    soundsD[sound] += 1
            exc_file.write(
                "### Variety {0} ({1}, {2})\n\n".format(
                    var, gcode["Glottocode"], " ".join(sources[var])
                )
            )
            exc_file.write("Sound in Source | BIPA \n")
            exc_file.write("--- | --- \n")
            for sound in vals:
                exc_file.write(
                    sound + " | " + dset_td.grapheme_map.get(sound, "?") + " \n"
                )
            count += 1
    exc_file.write("\n\n")
    exc_file.close()
    print("[i] excluded {0} inventories for {1}".format(count, dataset))
    print("missing gcodes: {0}".format(missing_gcodes))
    print("Problematic sounds: {0}".format(len(soundsD)))
    for s, count in sorted(soundsD.items(), key=lambda x: x[1]):
        print("{0:8} \t| {1}".format(s, count))

    with open(dataset + "-data.tsv", "w") as f:
        f.write(
            "\t".join(
                [
                    "ID",
                    "Name",
                    "Glottocode",
                    "Family",
                    "Macroarea",
                    "Latitude",
                    "Longitude",
                    "Sounds",
                    "Consonants",
                    "Vowels",
                    "Clusters",
                    "Diphthongs",
                    "LongConsonants",
                    "LongVowels",
                    "Ratio",
                    "Phonemes",
                ]
            )
            + "\n"
        )

        for inv in inventories.values():
            f.write(
                "\t".join(
                    [
                        inv.id,
                        inv.language.name,
                        inv.language.glottocode,
                        inv.language.family or "",
                        inv.language.macroarea or "",
                        str(inv.language.latitude or ""),
                        str(inv.language.longitude or ""),
                        str(len(inv.segments)),
                        str(len(inv.consonant_sounds)),
                        str(len(inv.vowel_sounds)),
                        str(len(inv.clusters)),
                        str(len(inv.diphthongs)),
                        str(long_consonants(inv)),
                        str(long_vowels(inv)),
                        str(len(inv.consonant_sounds) / len(inv.vowel_sounds)),
                        " ".join(inv.sounds),
                    ]
                )
                + "\n"
            )

    print("loaded {0} language varieties for {1}".format(count, dataset))
    stats = [
        len(varieties),
        len(varieties) - missing_gcodes,
        len(inventories),
        len(set([inv.language.glottocode for inv in inventories.values()])),
    ]
    return dump, stats


# load basic data
clts = CLTS()
bipa = clts.transcriptionsystem_dict["bipa"]

with open("output/excluded.md", "w") as f:
    f.write("# Excluded Varieties\n\n")

dump = defaultdict(list)
for ds in [
    "jipa",
    "UPSID",
    "lapsyd",
    "PHOIBLE",
    "AA",
    "RA",
    "SAPHON",
    "EA",
    "ER",
]:
    print("################################")
    print("# Importing data for {0}".format(ds))
    dump, (varieties, vars_with_gcode, inventories, distinct_gcodes) = load_dataset(
        ds, dump=dump
    )
    print("# Statistics on data in {0}".format(ds))
    print("- varieties:            {0}".format(varieties))
    print("- valid glottocodes:    {0}".format(vars_with_gcode))
    print("- inventories:          {0}".format(inventories))
    print("- distinct glottocodes: {0}".format(distinct_gcodes))

with open("app/data.js", "w") as f:
    f.write("var DATA = " + json.dumps(dump, indent=2) + ";\n")
