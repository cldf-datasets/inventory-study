import attr
from tqdm import tqdm


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

def progressbar(function, desc=''):

    return tqdm(function, desc=desc)



@attr.s
class Feature:
    """
    Feature class for the handling of linguistic features.
    """
    id = attr.ib()
    name = attr.ib(default=None)
    source = attr.ib(default=None)
    function = attr.ib(default=None)

    def __call__(self, *params, **kw):
        return self.function(*params, **kw)


wals_1a = Feature(
        id='WALS_1A',
        name='Consonant Inventory Size',
        source='Dryer2013',
        function=lambda inv: len(inv.consonants)
        )


wals_2a = Feature(
        id="WALS_2A",
        name="Vowel Inventory Size",
        source="Dryer2013",
        function=lambda inv: len(inv.vowels)
        )


wals_3a = Feature(
        id="WALS_2A",
        name="Consonant/Vowel Ratio",
        source="Dryer2013",
        function=lambda inv: len(inv.consonants) / len(inv.vowels)
        )




