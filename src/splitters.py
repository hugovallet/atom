"""
This modules define a base class and some specific Splitters. Splitters are used
in practice to split groups of Atoms to create new clusters.
"""
import abc
import collections
import copy
import itertools
from typing import List, Union, Tuple, Dict

import numpy as np
from sklearn.base import TransformerMixin, ClusterMixin
from sklearn.mixture._base import BaseMixture
from sklearn.pipeline import Pipeline
from typeguard import typechecked

from src.atoms import Atom, add_attribute_from_array

AnyNumber = Union[int, float]


class BaseSplitter:

    @typechecked
    def split(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        """
        The main method for splitting atom lists. It calls a child-class-defined `_split()` method
        and performs some input/output checks on the results to make sure that each child
        `_split()` returns something consistent with expectations. Each child `_split()` method
        should return a dictionary indexed by a cluster identifier (up to the coder to choose)
        and with list of atoms as values describing the content of the cluster. The concatenation
        of all the splits should return the original list of atoms (no atom creation /
        disappearance during split is allowed!)

        Returns:
            atom_groups: an iterable of lists of atoms

        Notes:
            This is the "design pattern" referred to as "Template Method Pattern". See
            https://en.wikipedia.org/wiki/Template_method_pattern
        """
        # 1. perform input checks
        assert isinstance(atom_list, list), f"Expecting parameter `atom_list` to be a list of " \
                                            f"Atom objects. You passed object of " \
                                            f"type {type(atom_list)}"
        assert isinstance(atom_list[0], Atom), f"Expecting parameter `atom_list` to be a list of " \
                                               f"{Atom} objects. You passed a list of " \
                                               f"type {type(atom_list[0])}"

        # 2. get split result from child method
        result = self._split(atom_list=atom_list)

        # 3. perform output checks
        # 3.a. check that concatenated list contains no duplicates
        concatenated_atom_list = list(itertools.chain.from_iterable(result.values()))
        unique_in_concatenated = set(concatenated_atom_list)
        assert len(unique_in_concatenated) == len(concatenated_atom_list), "Split result " \
                                                                           "contains atoms that " \
                                                                           "are in multiple " \
                                                                           "groups at the same " \
                                                                           "time: forbidden."
        # 3.b. check that concatenated list contains exactly the same elements as in original list
        assert unique_in_concatenated == set(atom_list), "Split result atoms are not the same as " \
                                                         "the ones in the input atom list: " \
                                                         "forbidden."

        # TODO: MAYBE: Assert that the number of customers in a certain cluster is
        # above a certain threshold. Maybe this should be done in the split function

        # 4. sort the dictionary keys alphabetically to ensure reproducibility
        sorted_result = {k: result[k] for k in sorted(result.keys())}

        return sorted_result

    @abc.abstractmethod
    def _split(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        """
        The method that really performs the split. *This method is specific to each splitter and
        should be overridden! It should not be called directly (thus it's private) but through
        a call to the base `split()` method above!*.

        Args:
            atom_list: the list of atoms to be split

        Returns:
            atom_groups: a dictionary of lists of atoms
        """
        pass

    def predict(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        """
        Method to predict atom clusters using a pre-fitted splitter. It calls a
        child-class-defined `_predict()` method and performs some input/output checks on the
        results to make sure that each child `_predict()` returns something consistent with
        expectations. Each child `_predict()` method should return a dictionary indexed by a cluster
        identifier (up to the coder to choose) and with list of atoms as values describing the
        content of the cluster.

        Args:
            atom_list: the list of atoms for which to predict cluster

        Returns:
            atom_groups: an dictionary of lists of atoms

        Notes:
            This is the "design pattern" referred to as "Template Method Pattern". See
            https://en.wikipedia.org/wiki/Template_method_pattern
        """
        # TODO: input checks: implement some
        result = self._predict(atom_list=atom_list)
        # TODO: output checks: implement some
        return result

    @abc.abstractmethod
    def _predict(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        """
        The method that really performs the predict. *This method is specific to each splitter and
        should be overridden! It should not be called directly (thus it's private) but through
        a call to the base `predict()` method above!*.

        Args:
            atom_list: the list of atoms to be split

        Returns:
            atom_groups: a dictionary of lists of atoms
        """
        pass

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        attr_str = ", ".join([f"{k[1:] if k.startswith('_') else k}={v}"
                              for k, v in self.__dict__.items()])
        if len(attr_str) > 50:
            attr_str = attr_str[:50] + "..."
        return f"<{self.__class__.__name__}> ({attr_str}) at {hex(id(self))}"


class AttrGreaterThan(BaseSplitter):
    """
    A simple splitter that puts all the atoms that `attr` equal `value` in a class an the rest in
    another one.
    """

    def __init__(self, attr: str, threshold: Union[int, float]):
        self._attr = attr
        self._threshold = threshold

    def _split(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        result = collections.defaultdict(list)
        for a in atom_list:
            if getattr(a, self._attr) > self._threshold:
                result[f"{self._attr}>{self._threshold}"].append(a)
            else:
                result[f"{self._attr}<={self._threshold}"].append(a)

        return result

    def _predict(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        return self.split(atom_list=atom_list)


class AttrEquals(BaseSplitter):
    """
    A simple splitter that puts all the atoms that `attr` equal `value` in a class an the rest in
    another one.
    """

    def __init__(self, attr: str, value: Union[int, float, str]):
        self._attr = attr
        self._value = value

    def _split(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        result = collections.defaultdict(list)
        for a in atom_list:
            if getattr(a, self._attr) == self._value:
                result[f"{self._attr}=={self._value}"].append(a)
            else:
                result[f"{self._attr}!={self._value}"].append(a)

        return result

    def _predict(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        return self.split(atom_list=atom_list)


class CategoricalSplitter(BaseSplitter):
    """
    A simple splitter that cuts the atoms based on the (discrete) value of `attr`.
    """

    def __init__(self, attr: str):
        self._attr = attr

    def _split(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        result = collections.defaultdict(list)
        for a in atom_list:
            val = getattr(a, self._attr)
            result[f"{self._attr}=={val}"].append(a)

        return result

    def _predict(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        return self.split(atom_list=atom_list)


class AttrMappingSplitter(BaseSplitter):
    """
    A splitter that simply takes a mapping ``{attribute_value: group}`` as a way to generate the
    groups. Values not in the mapping are affected to group "other"
    """

    def __init__(self, attr: str, attr_value_mapping: Dict[Union[str, int, float], str]):
        self._attr = attr
        self._attr_value_mapping = attr_value_mapping

    def _split(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        result = collections.defaultdict(list)
        for a in atom_list:
            val = getattr(a, self._attr)
            if val in self._attr_value_mapping.keys():
                group = self._attr_value_mapping[val]
                result[f"{group}"].append(a)
            else:
                result["other"].append(a)

        return result

    def _predict(self, atom_list: List[Atom]) -> Dict[Union[str, AnyNumber], List[Atom]]:
        return self.split(atom_list=atom_list)


class MLSplitter(BaseSplitter):
    """
    A Splitter that first performs a dimensionality reduction and then a clustering using SkLearn
    algorithms.

    Args:
        transformer_pipe: a list of SkLearn TransformerMixin objects to transform the baskets
        cluster_pipe: a list of clustering algorithm to get the clusters from the data transformed
                      by the transformer_pipe

    Returns:
        self: an instance of self
    """

    @typechecked
    def __init__(
            self,
            transformer_pipe: List[Tuple[str, TransformerMixin]],
            cluster_pipe: List[Tuple[str, Union[BaseMixture, ClusterMixin]]],
    ):
        self.transformer_pipe = transformer_pipe
        self.cluster_pipe = cluster_pipe

    @staticmethod
    def get_pca_component_name(component_number: int) -> str:
        """Return the PCA component_number-th component name used in this class

        Args:
            component_number: the index of the pca component
        Returns:
            the name used in this class for the pca component with index
            component_number
        """
        component_name = "component" + str(component_number)
        return component_name

    @staticmethod
    def _baskets_to_df(atom_list: List[Atom]):
        """
        Computes a numpy array from the basket features to be passed to SkLearn estimators

        Args:
            atom_list: atoms considered

        Returns:
            x: the features array
        """
        return np.array([a.basket_features for a in atom_list])

    def _split(self, atom_list: List[Atom]) -> Dict[str, List[Atom]]:
        """
        Splitting pipeline that performs 2 things in order to obtain new clusters
        for the passed atom_list:

        1. checks that all the atoms have a ``basket_features`` attribute. Put all the ones with
           null data in a separate cluster.
        2. on all the remaining ones (proper baskets):

            * apply the transformers described in ``self.transformer_pipe`` to the atom's baskets
            * apply the clustering described in ``self.transformer_pipe`` to get clusters

        Args:
            atom_list: list of atoms to segment

        Returns:
            result: the split atom list
        """
        result = collections.defaultdict(list)

        # set customers with nan baskets aside. If all basket are empty, do not try to cluster
        atom_list_nan = [a for a in atom_list if np.any(np.isnan(a.basket_features))]
        if len(atom_list_nan) > 0:
            result["nan"] = atom_list_nan
        atom_list_valid = [a for a in atom_list if a not in atom_list_nan]
        if len(atom_list_valid) == 0:
            return result

        # Reformat basket features into a numpy array for sklearn
        basket_features = self._baskets_to_df(atom_list_valid)

        # Run the transformer pipe
        pca_array = Pipeline(self.transformer_pipe).fit_transform(X=basket_features)

        # Run the clustering pipe
        new_clusters = Pipeline(self.cluster_pipe).fit_predict(X=pca_array)

        for i, cluster_id in enumerate(new_clusters):
            result[str(cluster_id)].append(atom_list_valid[i])

        # save transformed data into atoms for further retrieval
        add_attribute_from_array(atom_list_valid, attribute_array=pca_array,
                                 attribute_name=self.attribute_name)

        return result

    def _predict(self, atom_list: List[Atom]):
        """
        A method to predict the cluster the atoms fall into.

        Args:
            atom_list: list of atoms for which to predict the cluster

        Returns:
            result: an atom, cluster mapping
        """
        result = collections.defaultdict(list)

        # set customers with nan baskets aside. If all basket are empty, do not try to cluster
        atom_list_nan = [a for a in atom_list if np.any(np.isnan(a.basket_features))]
        if len(atom_list_nan) > 0:
            result["nan"] = atom_list_nan
        atom_list_valid = [a for a in atom_list if a not in atom_list_nan]
        if len(atom_list_valid) == 0:
            return result

        # Reformat basket features into a numpy array for sklearn
        basket_features = self._baskets_to_df(atom_list_valid)

        # Run the transformer pipe
        pca_array = Pipeline(self.transformer_pipe).transform(X=basket_features)

        # Run the clustering pipe
        new_clusters = Pipeline(self.cluster_pipe).predict(X=pca_array)

        for i, cluster_id in enumerate(new_clusters):
            result[str(cluster_id)].append(atom_list_valid[i])

        # save transformed data into atoms for further retrieval
        add_attribute_from_array(atom_list_valid, attribute_array=pca_array,
                                 attribute_name=self.attribute_name)

        return result

    @property
    def attribute_name(self):
        return f"projected_basket_feature_{hex(id(self))}"


AnySplitter = Union[BaseSplitter, AttrEquals, AttrGreaterThan, CategoricalSplitter, MLSplitter]
