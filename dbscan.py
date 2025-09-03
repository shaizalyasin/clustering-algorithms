import random

from typing import List

from classes.cluster import Cluster
from classes.point import Point


class DBSCAN:
    """A DBSCAN clustering instance."""

    def __init__(self, epsilon: float, min_pts: int) -> None:
        """
        Initialize the DBSCAN clustering instance.

        Parameters:
        epsilon (float): The maximum distance between two points to be considered neighbors.
        min_pts (int): The minimum number of points in a neighborhood.
        """
        # The maximum distance between two points to be considered neighbors
        self.epsilon: float = epsilon

        # The minimum number of points in a neighborhood
        self.min_pts: int = min_pts

        # The clusters (contrary to KMeans, we do not know the number of clusters in advance)
        self.clusters: List[Cluster] = []

        # A list of points that are identified as noise
        self.noise: List[Point] = []

    def fit(self, points: List[Point]) -> None:
        """
        Fit the DBSCAN clustering instance to the given points.

        Parameters:
        points (List[Point]): The points to cluster.
        """
        # TODO
        self.clusters = []
        self.noise = []

        for point in points:
            point.set_visited(False)
            point.set_clustered(False)

        for point in points:
            if point.is_visited():
                continue

            point.set_visited(True)
            neighborhood = self._get_neighborhood(point, points)

            if len(neighborhood) < self.min_pts:
                self.noise.append(point)
            else:
                new_cluster = Cluster()
                self.clusters.append(new_cluster)

                new_cluster.add_point(point)
                point.set_clustered(True)

                if point in self.noise:
                    self.noise.remove(point)

                queue_for_expansion = [p for p in neighborhood if p != point]

                while queue_for_expansion:
                    current_neighbor = queue_for_expansion.pop(0)  # Dequeue

                    if not current_neighbor.is_visited():
                        current_neighbor.set_visited(True)
                        neighbor_neighborhood = self._get_neighborhood(current_neighbor, points)

                        if len(neighbor_neighborhood) >= self.min_pts:
                            for p_in_nn in neighbor_neighborhood:
                                if not p_in_nn.is_visited():
                                    queue_for_expansion.append(p_in_nn)

                    if not current_neighbor.is_clustered():
                        new_cluster.add_point(current_neighbor)
                        current_neighbor.set_clustered(True)

                        if current_neighbor in self.noise:
                            self.noise.remove(current_neighbor)

    def _get_neighborhood(self, point: Point, points: List[Point]) -> List[Point]:
        """
        Get the neighborhood of a point.

        Parameters:
        point (Point): The point to get the neighborhood of.
        points (List[Point]): The points to consider.

        Returns:
        List[Point]: The points in the neighborhood.
        """
        # TODO
        neighborhood = []
        for other_point in points:
            distance = point.get_distance(other_point)

            if distance <= self.epsilon:
                neighborhood.append(other_point)
        return neighborhood
