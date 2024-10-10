"""
A module defining base class and shared methods to perform a segmentation on a set of Atoms
objects. Each problem defines its own segmentation tree by inheriting methods and attributes
from a BaseSegmentation class that contains all the tree-construction "back-end" logic. See section
"tutorials" for usage guidelines and examples.
"""
import abc
import collections
import logging
import time
from typing import List, Dict, Union

from src.analyses import SegmentationVisualizer
from src.atoms import Atom
from src.clusters import Cluster
from src.levels import Level
from src.misc import make_list
from src.splitters import AnySplitter


class BaseSegmentation:
    """
    Defines a segmentation tree. The tree is made of the following components:

        * atoms: the quantity we want to segment.
        * clusters: the groups we obtain by segmenting successively the initial list of atoms. The final segmentation is derived from the clusters we obtain on the lowest level.
        * levels: the different level of the segmentation, ``len(self.levels)`` = tree depth.

    There are basically 2 operations you can do in the Segmentation: splitting and grouping
    clusters of atoms. See specific methods documentation.

    Args:
        atoms: the list of atoms from which to build the segmentation tree

    Returns:
        self: an instance of self

    Notes:
        In order to maintain full history/transparency of the splitting/grouping process, whenever
        you use one of the 2 methods, a new level is appended to the tree and the result of the
        operation is stored on that new level, leaving the previous levels untouched.
    """

    def __init__(self, atoms: List[Atom]):
        self._log = logging.getLogger(__name__)

        # initialize tree structure
        self._atoms = atoms
        self._clusters = [Cluster(index=0, atom_list=atoms)]
        self._levels = [Level(index=0, cluster_list=self._clusters)]
        self._cluster_parents = collections.defaultdict(list)
        self._cluster_children = collections.defaultdict(list)
        self.links = dict()
        self.next = dict()
        self._history = []

    @property
    def levels(self) -> List[Level]:
        return self._levels

    @property
    def clusters(self) -> List[Cluster]:
        return self._clusters

    @property
    def atoms(self) -> List[Atom]:
        return self._atoms

    @property
    def lowest_clusters(self) -> List[Cluster]:
        """
        The current bottom clusters in tree

        Returns:
            The list of ids
        """
        return self.lowest_level.cluster_list

    @property
    def lowest_clusters_indices(self) -> List[int]:
        """
        The indices of the bottom clusters in tree

        Returns:
            The list of ids
        """
        return [c.index for c in self.lowest_clusters]

    @property
    def lowest_level(self) -> Level:
        return self.levels[-1]

    @property
    def cluster_children(self) -> Dict[Cluster, List[Cluster]]:
        return self._cluster_children

    @property
    def cluster_parents(self) -> Dict[Cluster, List[Cluster]]:
        return self._cluster_parents

    @property
    def history(self) -> List[Dict]:
        """
        An history to keep track of transforming operations such as grouping /
        splitting applied to the initial set of atoms. Useful when constructing
        the Segmentation "live" and that you've reached a satisfying result.
        """
        return self._history

    @property
    def plots(self) -> SegmentationVisualizer:
        """
        A pointer to the module generating visualizations.
        """
        return SegmentationVisualizer(self)

    def _get_cluster_index(self):
        return len(self._clusters)

    def _get_level_index(self):
        return len(self._levels)

    def split(self, splitter: AnySplitter, cluster_index: Union[int, List[int]] = None):
        """
        Splits the `cluster_index` using the splitter.

        Args:
            splitter: a Splitter object used to generate the split
            cluster_index: the index of list of indices of clusters to split. If the cluster_id is
                           omitted, the split will apply to all the clusters of the
                           lowest level of te current tree.
        """
        lowest_clusters = [c for c in self.lowest_clusters]

        if cluster_index is None:
            clusters_to_split = set(lowest_clusters)
        else:
            cluster_indices = make_list(cluster_index)
            clusters_to_split = {self.clusters[i] for i in cluster_indices}

        if not clusters_to_split.issubset(lowest_clusters):
            raise Exception("You tried splitting clusters that are not part of the lowest level:"
                            " operation forbidden.")

        t0 = time.time()
        self._log.info(f"Splitting on clusters {clusters_to_split} with splitter {splitter}")
        try:
            self._add_level(splitter=splitter)
            for cluster in lowest_clusters:
                atoms_in_cluster = cluster.atom_list
                if cluster in clusters_to_split:
                    # make a full copy of initial to store the splitter "fitted" on data
                    splitter_fitted = splitter.copy()
                    cluster.splitter = splitter_fitted
                    for decision_link, atom_group in splitter_fitted.split(
                            atoms_in_cluster).items():
                        new_cluster = Cluster(self._get_cluster_index(), atom_list=atom_group)
                        self._add_cluster(cluster=new_cluster)
                        self._add_link(cluster=new_cluster, parent_cluster=cluster,
                                       link=decision_link)
                else:
                    new_cluster = Cluster(self._get_cluster_index(), atom_list=atoms_in_cluster,
                                          name=cluster.name, local_name=cluster.local_name)
                    self._add_cluster(cluster=new_cluster)
                    self._add_link(cluster=new_cluster, parent_cluster=cluster, link="")
            self._history.append({"operation": "split", "splitter": splitter,
                                  "cluster_indices": [c.index for c in clusters_to_split]})
        except Exception as e:
            raise Exception(f"Error while splitting: {e}")
        finally:
            t1 = time.time()
            self._log.info(f"Done. ({t1 - t0: .2f}s)")

    def group(self, cluster_indices: List[int] = None):
        """
        Groups a list of clusters together.

        Args:
            cluster_indices: the list of clusters to merge together into 1.
        """
        lowest_clusters = [c for c in self.lowest_clusters]

        if cluster_indices is None:
            clusters_to_group = set(lowest_clusters)
        elif len(cluster_indices) >= 2:
            clusters_to_group = {self.clusters[i] for i in cluster_indices}
        else:
            raise Exception("You must pass at least 2 cluster indices to be grouped.")

        if any([c not in self.lowest_clusters_indices for c in cluster_indices]):
            raise Exception("You tried grouping clusters that are not part of the lowest level:"
                            " operation forbidden.")

        grouped_cluster = None
        t0 = time.time()
        self._log.info(f"Grouping on clusters {clusters_to_group}")
        try:
            self._add_level()
            for cluster in lowest_clusters:
                atoms_in_cluster = cluster.atom_list
                if cluster in clusters_to_group:
                    if grouped_cluster is None:
                        new_cluster = Cluster(self._get_cluster_index(), atom_list=atoms_in_cluster)
                        self._add_cluster(cluster=new_cluster)
                        self._add_link(cluster=new_cluster, parent_cluster=cluster, link="group")
                        grouped_cluster = new_cluster
                    else:
                        grouped_cluster._atom_list += atoms_in_cluster
                        self._add_link(cluster=grouped_cluster, parent_cluster=cluster,
                                       link="group")
                else:
                    new_cluster = Cluster(self._get_cluster_index(), atom_list=atoms_in_cluster,
                                          local_name=cluster.local_name, name=cluster.name)
                    self._add_cluster(cluster=new_cluster)
                    self._add_link(cluster=new_cluster, parent_cluster=cluster, link="")
            self._history.append({"operation": "group", "cluster_ids": clusters_to_group})
        except Exception as e:
            raise Exception(f"Error while grouping: {e}")
        finally:
            t1 = time.time()
            self._log.info(f"Done. ({t1 - t0: .2f}s)")

    def predict(self, atom: Atom, current_cluster=None) -> Cluster:
        if current_cluster is None:
            current_cluster = self.clusters[0]
        try:
            if len(current_cluster.children) == 0:
                return current_cluster
            elif len(current_cluster.children) == 1:
                next_cluster = current_cluster.children[0]
                return self.predict(atom=atom, current_cluster=next_cluster)
            else:
                decision = list(current_cluster.splitter.predict([atom]).keys())[0]
                next_cluster = self.next[(current_cluster, decision)]
                return self.predict(atom=atom, current_cluster=next_cluster)

        except Exception as e:
            raise Exception(f"Error happened while predicting cluster for atom {atom}: {e}")

    def _add_level(self, splitter: AnySplitter = None):
        new_max_level = Level(index=self._get_level_index(), cluster_list=[], splitter=splitter)
        self._levels.append(new_max_level)

    def _add_cluster(self, cluster: Cluster):
        self._clusters.append(cluster)
        self.lowest_level.cluster_list.append(cluster)

    def _add_link(self, cluster: Cluster, parent_cluster: Cluster, link):
        self._cluster_parents[cluster].append(parent_cluster)
        cluster.parents.append(parent_cluster)
        self._cluster_children[parent_cluster].append(cluster)
        parent_cluster.children.append(cluster)
        self.links[(parent_cluster, cluster)] = link
        self.next[(parent_cluster, link)] = cluster

    @abc.abstractmethod
    def run(self):
        """
        A method to build the full segmentation Tree. Needs to be overridden by each segmentation
        problem to define its own structure.
        """
        pass

    def __getstate__(self) -> dict:
        """
        Make the Segmentation serializable by removing the non-serializable logger.

        Returns:
            state: the dictionary containing object attributes
        """
        state = self.__dict__.copy()
        del state['_log']
        return state

    def __setstate__(self, state: dict):
        """
        Make the Segmentation deserializable by adding the non-serialized logger.

        Args:
            state: deserialized dictionary of attributes
        """
        self.__dict__.update(state)
        self._log = logging.getLogger(__name__)


class CustomSegmentation(BaseSegmentation):
    """
    This is the class template to create a custom segmentation for a given problem.
    """

    def __init__(self, atoms: List[Atom]):
        super().__init__(atoms)

    def run(self):
        raise NotImplementedError("Todo: this needs to be ported from the old segmentation module")


AnySegmentation = Union[BaseSegmentation, CustomSegmentation]
