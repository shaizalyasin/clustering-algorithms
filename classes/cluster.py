from typing import List

from classes.point import Point


class Cluster:
    """
    A cluster of points.

    Can also be used as a partition if a partition-based clustering algorithm is implemented.
    """

    def __init__(self):
        """
        Initialize the cluster.
        """
        # The points in the cluster
        self.points = []

    def add_point(self, point: Point):
        """
        Add a point to the cluster.

        Parameters:
        point (Point): The point to add to the cluster
        """
        self.points.append(point)

    def remove_point(self, point: Point):
        """
        Remove a point from the cluster.
        """
        self.points.remove(point)

    def get_points(self) -> List[Point]:
        """
        Get all points in the cluster.

        Returns:
        List[Point]: A list of all points in the cluster
        """
        return self.points

    def __contains__(self, point: Point) -> bool:
        """
        Check if a point is in the cluster.
        """
        return point in self.points

    def __len__(self):
        """
        Get the number of points in the cluster.
        """
        return len(self.points)

    def __iter__(self):
        """
        Iterate over all points in the cluster.
        """
        return iter(self.points)

    def __str__(self):
        """
        Return a string representation of the cluster.
        """
        return str(self.points)

    def __repr__(self):
        """
        Return a string representation of the cluster.
        """
        return self.__str__()
