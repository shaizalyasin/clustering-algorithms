from kmeans import KMeans

from classes.point import Point


#####
# Test(s) with the small dataset of 2D points
#####


def test_with_small_point_dataset_and_two_partitions(small_point_dataset):
    """Test fitting the K-Means instance with the small dataset of 2D points and 2 partitions."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Fit the K-Means instance
    kmeans.fit(small_point_dataset)

    # As the initialization of the partitions is not defined by us, there might be different results
    # However, we can check that:

    # 1) Both partitions are not empty
    assert len(kmeans.partitions[0]) > 0, "The first partition is empty."
    assert len(kmeans.partitions[1]) > 0, "The second partition is empty."

    # 2) The count of points in all partitions is equal to the count of points in the dataset
    # = There are no extra points in the partitions
    assert len(kmeans.partitions[0]) + len(kmeans.partitions[1]) == len(
        small_point_dataset
    ), f"The count of points in all partitions is not equal to the count of points in the dataset. The count of points in all partitions is {len(kmeans.partitions[0]) + len(kmeans.partitions[1])} and the count of points in the dataset is {len(small_point_dataset)}."

    # 3) Each point is in one of the partitions
    for point in small_point_dataset:
        assert (
            point in kmeans.partitions[0] or point in kmeans.partitions[1]
        ), f"The point {point} is not in any partition."

    # 4) Each point is closest to the centroid of its own partition (end criterion of the K-Means algorithm)
    # (We calculate the centroids of each partition by ourselves in this test case, as the centroids stored in the kmeans object might be incorrect, if _update_centroids() is not implemented correctly)
    centroids = []
    for partition in kmeans.partitions:
        # Calculate the centroid of each partition
        points = partition.get_points()
        mean_x = sum([point.x for point in points]) / len(points)
        mean_y = sum([point.y for point in points]) / len(points)
        centroids.append(Point(mean_x, mean_y))

    for i, partition in enumerate(kmeans.partitions):
        for point in partition:
            # Get the distances to all centroids
            distances = [point.get_distance(centroid) for centroid in centroids]

            # Assert that the distance to the centroid of its own partition is the smallest
            assert distances[i] == min(
                distances
            ), f"As there are centroids closer to point {point} than the centroid of its own partition, the point {point} should be in another partition and therefore k-means should not have converged yet."


def test_with_small_point_dataset_and_three_partitions(small_point_dataset):
    """Test fitting the K-Means instance with the small dataset of 2D points and 3 partitions."""

    # Create a K-Means instance with 3 clusters
    kmeans = KMeans(3)

    # Fit the K-Means instance
    kmeans.fit(small_point_dataset)

    # As the initialization of the partitions is not defined by us, there might be different results
    # However, we can check that:

    # 1) All partitions are not empty
    assert len(kmeans.partitions[0]) > 0, "The first partition is empty."
    assert len(kmeans.partitions[1]) > 0, "The second partition is empty."
    assert len(kmeans.partitions[2]) > 0, "The third partition is empty."

    # 2) The count of points in all partitions is equal to the count of points in the dataset
    # = There are no extra points in the partitions
    assert len(kmeans.partitions[0]) + len(kmeans.partitions[1]) + len(
        kmeans.partitions[2]
    ) == len(
        small_point_dataset
    ), f"The count of points in all partitions is not equal to the count of points in the dataset. The count of points in all partitions is {len(kmeans.partitions[0]) + len(kmeans.partitions[1]) + len(kmeans.partitions[2])} and the count of points in the dataset is {len(small_point_dataset)}."

    # 3) Each point is in one of the partitions
    for point in small_point_dataset:
        assert (
            point in kmeans.partitions[0]
            or point in kmeans.partitions[1]
            or point in kmeans.partitions[2]
        ), f"The point {point} is not in any partition."

    # 4) Each point is closest to the centroid of its own partition (end criterion of the K-Means algorithm)
    # (We calculate the centroids of each partition by ourselves in this test case, as the centroids stored in the kmeans object might be incorrect, if _update_centroids() is not implemented correctly)
    centroids = []
    for partition in kmeans.partitions:
        # Calculate the centroid of each partition
        points = partition.get_points()
        mean_x = sum([point.x for point in points]) / len(points)
        mean_y = sum([point.y for point in points]) / len(points)
        centroids.append(Point(mean_x, mean_y))

    for i, partition in enumerate(kmeans.partitions):
        for point in partition:
            # Get the distances to all centroids
            distances = [point.get_distance(centroid) for centroid in centroids]

            # Assert that the distance to the centroid of its own partition is the smallest
            assert distances[i] == min(
                distances
            ), f"As there are centroids closer to point {point} than the centroid of its own partition, the point {point} should be in another partition and therefore k-means should not have converged yet."


#####
# Test(s) with the bigger dataset of 2D points
#####


def test_with_bigger_point_dataset_and_two_partitions(bigger_point_dataset):
    """Test fitting the K-Means instance with the bigger dataset of 2D points and 2 partitions."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Fit the K-Means instance
    kmeans.fit(bigger_point_dataset)

    # As the initialization of the partitions is not defined by us, there might be different results
    # However, we can check that:

    # 1) Both partitions are not empty
    assert len(kmeans.partitions[0]) > 0, "The first partition is empty."
    assert len(kmeans.partitions[1]) > 0, "The second partition is empty."

    # 2) The count of points in all partitions is equal to the count of points in the dataset
    # = There are no extra or missing points in the partitions
    assert len(kmeans.partitions[0]) + len(kmeans.partitions[1]) == len(
        bigger_point_dataset
    ), f"The count of points in all partitions is not equal to the count of points in the dataset. The count of points in all partitions is {len(kmeans.partitions[0]) + len(kmeans.partitions[1])} and the count of points in the dataset is {len(bigger_point_dataset)}."

    # 3) Each point is in one of the partitions
    # (Since (5,5) is in the dataset twice, this is not really a 100% safe check to ensure that each point is in one of the partitions, but together with 2) it is close enough)
    for point in bigger_point_dataset:
        assert (
            point in kmeans.partitions[0] or point in kmeans.partitions[1]
        ), f"The point {point} is not in any partition."

    # 4) Each point is closest to the centroid of its own partition (end criterion of the K-Means algorithm)
    # (We calculate the centroids of each partition by ourselves in this test case, as the centroids stored in the kmeans object might be incorrect, if _update_centroids() is not implemented correctly)
    centroids = []
    for partition in kmeans.partitions:
        # Calculate the centroid of each partition
        points = partition.get_points()
        mean_x = sum([point.x for point in points]) / len(points)
        mean_y = sum([point.y for point in points]) / len(points)
        centroids.append(Point(mean_x, mean_y))

    for i, partition in enumerate(kmeans.partitions):
        for point in partition:
            # Get the distances to all centroids
            distances = [point.get_distance(centroid) for centroid in centroids]

            # Assert that the distance to the centroid of its own partition is the smallest
            assert distances[i] == min(
                distances
            ), f"As there are centroids closer to point {point} than the centroid of its own partition, the point {point} should be in another partition and therefore k-means should not have converged yet."
