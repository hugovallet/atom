"""
A module to define the most granular level of the segmentation: Atoms.
"""
import logging
import time
from typing import List, Union

import numpy as np
import pandas as pd

from src.misc import reduce_list


class Atom(object):
    """
    Class defining a segmentation Atom. An atom is the quantity that is being segmented. In
    practice, it can be used to define a customer, a product, a sku, etc.
    """

    def __init__(self, index: int, id: Union[int, str], *args, **kwargs):
        self.index = index
        self.id = id
        for k, val in kwargs.items():
            try:
                setattr(self, k, val)
            except AttributeError as e:
                raise AttributeError(f"Could not set Atom attribute '{k}': {e}")

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)

    def __repr__(self):
        return f"<{self.__class__.__name__}> (index={self.index}) at {hex(id(self))}"


class FastAtom(Atom):
    """
    A lightweight Atom object that assumes a certain data structure in the input and take advantage
    of it leveraging __slots__ to make object instantiation faster. Can be useful for large databases.
    """

    def __init__(self, *args):
        object.__init__(self)
        for attr_name, value in zip(self.__slots__, args):
            setattr(self, attr_name, value)


def generate_atoms_from_df(customer_data_cube_df: pd.DataFrame) -> List[Atom]:
    """
    Generates the list of atoms from the customer DataFrame.

    Args:
        customer_data_cube_df: a data cube with id as index and attributes as columns

    Returns:
        atoms_list: the list of atoms objects
    """
    logging.info(f"Generating {len(customer_data_cube_df)} atoms from DataFrame "
                 f"{customer_data_cube_df.shape}")
    t0 = time.time()
    dct = customer_data_cube_df.to_dict(orient="index")
    atoms_list = [Atom(index=index, id=id, **args) for index, (id, args) in
                  enumerate(dct.items())]
    t1 = time.time()
    logging.info(f"Done. ({t1 - t0: .2f}s)")
    return atoms_list


def fast_generate_atoms_from_df(customer_data_cube_df: pd.DataFrame) -> List[FastAtom]:
    """
    Same as function above but uses a specific optimized Atom to gain speed.

    Args:
        customer_data_cube_df:  a data cube with id as index and attributes as columns

    Returns:
        atoms_list: the list of fast atoms objects
    """
    logging.info(f"Generating {len(customer_data_cube_df)} atoms from DataFrame "
                 f"{customer_data_cube_df.shape}")

    required_columns = FastAtom.__slots__[2:]
    missing = set(required_columns) - set(customer_data_cube_df.columns)
    if len(missing) > 0:
        raise Exception(f"Cannot generate atoms, missing columns in input data: "
                        f"{missing}")

    t0 = time.time()
    atoms_list = [FastAtom(index, *args) for index, args in
                  enumerate(customer_data_cube_df.itertuples(index=True))]
    t1 = time.time()
    logging.info(f"Done. ({t1 - t0: .2f}s)")
    return atoms_list


def generate_df_from_atoms(atoms: List[Atom]) -> pd.DataFrame:
    """
    Opposite of function above. Please note that this will note convert atom's attribute that are
    not uni-dimensional (typically baskets).

    Args:
        atoms: the list of atoms to convert to DataFrame

    Returns:
        df: the DataFrame equivalent of the list of atoms
    """
    res = dict()

    uni_dimensional = [att for att, val in atoms[0].__dict__.items() if
                       type(val) in [int, str, float] and att != "id" and not att.startswith("_")]

    for a in atoms:
        res[a.id] = {key: getattr(a, key) for key in uni_dimensional}

    df = pd.DataFrame.from_dict(res, orient="index")
    return df


def add_attribute_from_df(atoms_list: List[Atom], attribute_df: pd.DataFrame, attribute_name: str,
                          null_values: str = "raise"):
    """
    Adds an attribute from a DataFrame to a list of atoms. Typically used to link the basket
    features.

    Args:
        atoms_list: List of atoms
        attribute_df: consisting of the the attribute value for each atom. Index must be set on the
                      Atom local id, for instance "customer_id".
        attribute_name: Name of the new attribute. The indices here have to
                        correspond to the customer
        null_values: policy to apply when managing null values
    """
    # TODO: check index is set correctly

    # create atom index
    id_to_atom = {a.id: a for a in atoms_list}

    # reindex attribute_df to it
    attribute_df = attribute_df.reindex(id_to_atom.keys())

    # manage NAs
    for col, count in attribute_df.isnull().sum(axis=0).to_dict().items():
        if count > 0 and null_values == "raise":
            raise ValueError(f"Column {col} contains {count} Nan: can't add attribute")
        if count > 0 and null_values == "warn":
            logging.warning(f"Column {col} contains {count} Nan")

    try:
        for row in attribute_df.itertuples(index=True):
            setattr(id_to_atom[row[0]], attribute_name, row[1:])
        return atoms_list

    except Exception as e:
        raise e


def add_attribute_from_array(atoms_list: List[Atom], attribute_array: np.array,
                             attribute_name: str):
    """
    Adds an attribute from a numpy 2D Array a list of atoms. Typically used to relink the basket
    features after their transformation by a dimensionality reduction algorithm (such as PCA).

    Args:
        atoms_list:  List of atoms
        attribute_array: consisting of the the attribute value for each atom
        attribute_name: Name of the new attribute. The indices here have to
                        correspond to the customer
    """
    if attribute_array.shape[0] != len(atoms_list):
        raise Exception("Number of lines in the array not aligned with number of atoms: forbidden.")

    try:
        for i, row in enumerate(attribute_array):
            setattr(atoms_list[i], attribute_name, tuple(row))
    except Exception as e:
        raise e


def atom_attr_argmax(atoms_list: List[Atom], attribute_name: str, n: int = None) -> \
        Union[Atom, List[Atom]]:
    """
    Selects the `n`atoms in `atoms_list` which have the largest value for attribute
    `attribute_name`.

    Args:
        atoms_list: the list of atoms in which to lookup
        attribute_name: the attribute to maximize
        n: the number of top atoms to return

    Returns:
        sorted_list: the list of atoms object from top `attribute_name` to least
    """
    sorted_list = sorted(atoms_list, key=lambda x: getattr(x, attribute_name), reverse=True)
    if n is not None:
        return reduce_list(sorted_list[:n])
    else:
        return reduce_list(sorted_list)


def atom_attr_argmin(atoms_list: List[Atom], attribute_name: str, n: int = None) -> \
        Union[Atom, List[Atom]]:
    """
    Selects the `n`atoms in `atoms_list` which have the lowest value for attribute
    `attribute_name`.

    Args:
        atoms_list: the list of atoms in which to lookup
        attribute_name: the attribute to maximize
        n: the number of top atoms to return

    Returns:
        sorted_list: the list of atoms object from top `attribute_name` to least
    """
    sorted_list = sorted(atoms_list, key=lambda x: getattr(x, attribute_name), reverse=False)
    if n is not None:
        return reduce_list(sorted_list[:n])
    else:
        return reduce_list(sorted_list)
