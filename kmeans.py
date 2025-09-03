import random

from typing import List

from classes.point import Point
from classes.cluster import Cluster


class KMeans:
    """
    A K-Means clustering instance.
    """

    def __init__(self, k: int):
        """
        Initialize the K-Means clustering instance.

        Parameters:
        k (int): The number of clusters to create
        """
        # The number of partitions/clusters to create
        self.k: int = k

        # Create the partitions/clusters
        self.partitions: List[Cluster] = [Cluster() for _ in range(k)]

        # The centroids of the partitions
        self.centroids: List[Point] = [None for _ in range(k)]

    def fit(self, points: List[Point]):
        """
        Fit the K-Means clustering instance to the given points.

        Parameters:
        points (List[Point]): The points to cluster
        """
        # TODO
        # common heuristic value for k-means
        max_iterations = 300
        self._initialize_partitions(points)

        for i in range(max_iterations):
            self._update_centroids()
            partitions_changed = self._reassign_points()

            if not partitions_changed:
                break

    def _initialize_partitions(self, points: List[Point]):
        """
        Initializes the partitions (self.partitions) by assigning each point to a cluster/partition.
        All clusters/partitions are non empty after this method is called.

        Parameters:
        points (List[Point]): The points to cluster
        """
        # Check if the number of points is greater or equal to the number of clusters
        if len(points) < self.k:
            raise ValueError(
                "Number of points has to be greater or equal to the number of clusters."
            )

        # TODO
        for partition in self.partitions:
            partition.points.clear()

        for i in range(self.k):
            self.partitions[i].add_point(points[i])

        for i in range(self.k, len(points)):
            random_partition_index = random.randint(0, self.k - 1)
            self.partitions[random_partition_index].add_point(points[i])

    def _update_centroids(self):
        """
        Updates the centroids of the partitions and writes the new centroids into self.centroids.
        """
        # Make sure that all partitions are non-empty
        for partition in self.partitions:
            if len(partition) == 0:
                # Throw an error if a partition is empty
                raise ValueError(
                    "All partitions have to be non-empty before updating the centroids."
                )

        # TODO
        for i, partition in enumerate(self.partitions):
            points_in_partition = partition.get_points()

            sum_x = 0.0
            sum_y = 0.0

            for point in points_in_partition:
                sum_x += point.get_x()
                sum_y += point.get_y()

            num_points = len(points_in_partition)

            new_centroid_x = sum_x / num_points
            new_centroid_y = sum_y / num_points

            self.centroids[i] = Point(new_centroid_x, new_centroid_y)

    def _reassign_points(self) -> bool:
        """
        Reassigns each point to the partition with the closest centroid.

        Ensures that each partition is non-empty after reassigning the points,
        by randomly reassigning a single point from a random non-empty partition
        with more than one element into each empty partition.

        This is necessary to avoid empty partitions, which would lead to k-means
        not producing k clusters, but k-n clusters (n is the number of empty partitions).

        Returns:
        bool: True if the reassignment changed the partitions, False otherwise
        """
        # TODO
        partitions_changed = False
        all_points_with_current_cluster_idx: List[Point] = []

        for i, partition in enumerate(self.partitions):
            for point in partition.get_points():
                all_points_with_current_cluster_idx.append((point, i))

        new_partitions: List[Cluster] = [Cluster() for _ in range(self.k)]

        for point, old_partition_idx in all_points_with_current_cluster_idx:
            closest_centroid_distance = float('inf')
            closest_centroid_idx = -1

            for j, centroid in enumerate(self.centroids):
                distance = point.get_distance(centroid)

                if distance < closest_centroid_distance:
                    closest_centroid_distance = distance
                    closest_centroid_idx = j
                elif distance == closest_centroid_distance:
                    if j == old_partition_idx:
                        closest_centroid_idx = j

            new_partitions[closest_centroid_idx].add_point(point)

            if closest_centroid_idx != old_partition_idx:
                partitions_changed = True

        self.partitions = new_partitions

        empty_partition_indices = [i for i, p in enumerate(self.partitions) if len(p) == 0]

        donor_partition_indices = [
            i for i, p in enumerate(self.partitions) if len(p) > 1
        ]

        random.shuffle(donor_partition_indices)

        for empty_idx in empty_partition_indices:
            if not donor_partition_indices:
                continue

            donor_idx = donor_partition_indices.pop(0)
            donor_partition = self.partitions[donor_idx]

            point_to_move = random.choice(donor_partition.get_points())
            donor_partition.remove_point(point_to_move)

            self.partitions[empty_idx].add_point(point_to_move)
            partitions_changed = True

            if len(donor_partition) <= 1:
                donor_partition_indices = [
                    i for i, p in enumerate(self.partitions) if len(p) > 1
                ]
                random.shuffle(donor_partition_indices)

        return partitions_changed
