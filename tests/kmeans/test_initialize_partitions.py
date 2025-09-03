from kmeans import KMeans

#####
# Test(s) with the small dataset of 2D points
#####


def test_with_small_point_dataset_and_two_partitions(small_point_dataset):
    """Test the initialization of the partitions with the small dataset of 2D points and 2 partitions."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Initialize the partitions
    kmeans._initialize_partitions(small_point_dataset)

    # Check that each partition contains at least one point
    assert len(kmeans.partitions[0]) > 0, "The first partition is empty."
    assert len(kmeans.partitions[1]) > 0, "The second partition is empty."

    # Check that the count of points in all partitions is equal to the count of points in the dataset
    assert len(kmeans.partitions[0]) + len(kmeans.partitions[1]) == len(
        small_point_dataset
    ), f"The count of points in all partitions is not equal to the count of points in the dataset. The count of points in all partitions is {len(kmeans.partitions[0]) + len(kmeans.partitions[1])} and the count of points in the dataset is {len(small_point_dataset)}."

    # Check that each point is in one of the partitions
    for point in small_point_dataset:
        assert (
            point in kmeans.partitions[0] or point in kmeans.partitions[1]
        ), f"The point {point} is not in any partition."


def test_with_small_point_dataset_and_three_partitions(small_point_dataset):
    """Test the initialization of the partitions with the small dataset of 2D points and 3 partitions."""

    # Create a K-Means instance with 3 clusters
    kmeans = KMeans(3)

    # Initialize the partitions
    kmeans._initialize_partitions(small_point_dataset)

    # Check that each partition contains at least one point
    assert len(kmeans.partitions[0]) > 0, "The first partition is empty."
    assert len(kmeans.partitions[1]) > 0, "The second partition is empty."
    assert len(kmeans.partitions[2]) > 0, "The third partition is empty."

    # Check that the count of points in all partitions is equal to the count of points in the dataset
    assert len(kmeans.partitions[0]) + len(kmeans.partitions[1]) + len(
        kmeans.partitions[2]
    ) == len(
        small_point_dataset
    ), f"The count of points in all partitions is not equal to the count of points in the dataset. The count of points in all partitions is {len(kmeans.partitions[0]) + len(kmeans.partitions[1]) + len(kmeans.partitions[2])} and the count of points in the dataset is {len(small_point_dataset)}."

    # Check that each point is in one of the partitions
    for point in small_point_dataset:
        assert (
            point in kmeans.partitions[0]
            or point in kmeans.partitions[1]
            or point in kmeans.partitions[2]
        ), f"The point {point} is not in any partition."
