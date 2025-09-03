class Point:
    """A class representing a point in a 2D space"""

    def __init__(
        self, x: float, y: float, visited: bool = False, clustered: bool = False
    ):
        """
        Initialize the point

        Parameters:
        x (float): The x-coordinate of the point
        y (float): The y-coordinate of the point
        visited (bool): A flag for DBSCAN to indicate if the point has been visited, default: False
        clustered (bool): A flag for DBSCAN to indicate if the point is part of a cluster, default: False
        """
        # The x-coordinate of the point
        self.x = x

        # The y-coordinate of the point
        self.y = y

        # A flag to indicate if the point has been visited
        # (should be used in DBSCAN, but is not used in KMeans)
        self.visited = visited

        # A flag to indicate if the point is part of a cluster
        # (should be used in DBSCAN, but is not used in KMeans)
        self.clustered = clustered

    def get_x(self) -> float:
        """
        Get the x-coordinate of the point

        Returns:
        float: The x-coordinate of the point
        """

        return self.x

    def get_y(self) -> float:
        """
        Get the y-coordinate of the point

        Returns:
        float: The y-coordinate of the point
        """

        return self.y

    def get_visited(self) -> bool:
        """
        Get the visited flag of the point

        Returns:
        bool: The visited flag of the point
        """

        return self.visited

    def get_clustered(self) -> bool:
        """
        Get the clustered flag of the point

        Returns:
        bool: The clustered flag of the point
        """

        return self.clustered

    def get_distance(self, other) -> float:
        """
        Get the Euclidean distance between this point and another point

        Parameters:
        other (Point): The other point to calculate the distance to

        Returns:
        float: The Euclidean distance between this point and the other point
        """

        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def is_visited(self) -> bool:
        """
        Check if the point has been visited

        Returns:
        bool: Whether the point has been visited
        """

        return self.visited

    def is_clustered(self) -> bool:
        """
        Check if the point is part of a cluster

        Returns:
        bool: Whether the point is part of a cluster
        """

        return self.clustered

    def set_visited(self, visited: bool = True) -> None:
        """
        Set the visited flag of the point

        Parameters:
        visited (bool): The visited flag to set, default: True
        """

        self.visited = visited

    def set_clustered(self, clustered: bool = True) -> None:
        """
        Set the clustered flag of the point

        Parameters:
        clustered (bool): The clustered flag to set, default: True
        """

        self.clustered = clustered

    def __str__(self, additional_info: bool = False) -> str:
        """
        Get a string representation of the point

        Parameters:
        additional_info (bool): A flag to indicate if additional information should be included, default: False

        Returns:
        str: A string representation of the point
        """

        # Include additional information if the info flag is set
        if additional_info:
            if self.visited:
                if self.clustered:
                    return f"({self.x}, {self.y}) - clustered, visited"
                else:
                    return f"({self.x}, {self.y}) - not clustered, visited"
            else:
                if self.clustered:
                    return f"({self.x}, {self.y}) - clustered, not visited"
                else:
                    return f"({self.x}, {self.y}) - not clustered, not visited"

        # Otherwise, just return the coordinates
        return f"({self.x}, {self.y})"

    def __repr__(self, additional_info: bool = False) -> str:
        """
        Get a string representation of the point

        Parameters:
        additional_info (bool): A flag to indicate if additional information should be included, default: False

        Returns:
        str: A string representation of the point
        """

        return self.__str__(additional_info)

    def __eq__(self, other) -> bool:
        """
        Check if the point is equal to another point

        Parameters:
        other (Point): The other point to compare

        Returns:
        bool: True if the point is equal to the other point, False otherwise
        """

        return self.x == other.x and self.y == other.y
