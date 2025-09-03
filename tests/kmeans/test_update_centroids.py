import pytest

from kmeans import KMeans

#####
# Test(s) with the small dataset of 2D points
#####


def test_with_small_point_dataset_and_two_partitions(small_point_dataset):
    """Test the update of the centroids with the small dataset of 2D points and 2 partitions."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Initialize the partitions (manually, to be able to only test _update_centroids())
    kmeans.partitions[0].add_point(small_point_dataset[0])  # (1,1)
    kmeans.partitions[0].add_point(small_point_dataset[1])  # (1,2)
    kmeans.partitions[0].add_point(small_point_dataset[2])  # (1,4)
    kmeans.partitions[1].add_point(small_point_dataset[3])  # (2,1)
    kmeans.partitions[1].add_point(small_point_dataset[4])  # (2,3)
    kmeans.partitions[1].add_point(small_point_dataset[5])  # (3,2)
    kmeans.partitions[1].add_point(small_point_dataset[6])  # (3,4)
    kmeans.partitions[1].add_point(small_point_dataset[7])  # (4,1)
    kmeans.partitions[1].add_point(small_point_dataset[8])  # (4,3)
    kmeans.partitions[1].add_point(small_point_dataset[9])  # (4,4)

    # Update the centroids
    kmeans._update_centroids()

    # Check that the centroids are updated correctly
    assert kmeans.centroids[0].get_x() == pytest.approx(
        1
    ), f"The x coordinate of the first centroid is not correct. The x coordinate of the first centroid is {kmeans.centroids[0].get_x()} and it should be 1."
    assert kmeans.centroids[0].get_y() == pytest.approx(
        2.333333
    ), f"The y coordinate of the first centroid is not correct. The y coordinate of the first centroid is {kmeans.centroids[0].get_y()} and it should be 2.333333."

    assert kmeans.centroids[1].get_x() == pytest.approx(
        3.142857
    ), f"The x coordinate of the second centroid is not correct. The x coordinate of the second centroid is {kmeans.centroids[1].get_x()} and it should be 3.142857."
    assert kmeans.centroids[1].get_y() == pytest.approx(
        2.571429
    ), f"The y coordinate of the second centroid is not correct. The y coordinate of the second centroid is {kmeans.centroids[1].get_y()} and it should be 2.571429."


def test_with_small_point_dataset_and_three_partitions(small_point_dataset):
    """Test the update of the centroids with the small dataset of 2D points and 3 partitions."""

    # Create a K-Means instance with 3 clusters
    kmeans = KMeans(3)

    # Initialize the partitions (manually, to be able to only test _update_centroids())
    kmeans.partitions[0].add_point(small_point_dataset[0])  # (1,1)
    kmeans.partitions[0].add_point(small_point_dataset[1])  # (1,2)
    kmeans.partitions[0].add_point(small_point_dataset[3])  # (2,1)
    kmeans.partitions[1].add_point(small_point_dataset[2])  # (1,4)
    kmeans.partitions[1].add_point(small_point_dataset[4])  # (2,3)
    kmeans.partitions[1].add_point(small_point_dataset[5])  # (3,2)
    kmeans.partitions[1].add_point(small_point_dataset[6])  # (3,4)
    kmeans.partitions[2].add_point(small_point_dataset[7])  # (4,1)
    kmeans.partitions[2].add_point(small_point_dataset[8])  # (4,3)
    kmeans.partitions[2].add_point(small_point_dataset[9])  # (4,4)

    # Update the centroids
    kmeans._update_centroids()

    # Check that the centroids are updated correctly
    assert kmeans.centroids[0].get_x() == pytest.approx(
        1.333333
    ), f"The x coordinate of the first centroid is not correct. The x coordinate of the first centroid is {kmeans.centroids[0].get_x()} and it should be 1.333333."
    assert kmeans.centroids[0].get_y() == pytest.approx(
        1.333333
    ), f"The y coordinate of the first centroid is not correct. The y coordinate of the first centroid is {kmeans.centroids[0].get_y()} and it should be 1.333333."

    assert kmeans.centroids[1].get_x() == pytest.approx(
        2.25
    ), f"The x coordinate of the second centroid is not correct. The x coordinate of the second centroid is {kmeans.centroids[1].get_x()} and it should be 2.25."
    assert kmeans.centroids[1].get_y() == pytest.approx(
        3.25
    ), f"The y coordinate of the second centroid is not correct. The y coordinate of the second centroid is {kmeans.centroids[1].get_y()} and it should be 3.25."

    assert kmeans.centroids[2].get_x() == pytest.approx(
        4
    ), f"The x coordinate of the third centroid is not correct. The x coordinate of the third centroid is {kmeans.centroids[2].get_x()} and it should be 4."
    assert kmeans.centroids[2].get_y() == pytest.approx(
        2.666667
    ), f"The y coordinate of the third centroid is not correct. The y coordinate of the third centroid is {kmeans.centroids[2].get_y()} and it should be 2.666667."


def test_with_small_point_dataset_and_four_partitions(small_point_dataset):
    """Test the update of the centroids with the small dataset of 2D points and 4 partitions (three of them only containing a single point)."""

    # Create a K-Means instance with 4 clusters
    kmeans = KMeans(4)

    # Initialize the partitions (manually, to be able to only test _update_centroids())
    kmeans.partitions[0].add_point(small_point_dataset[0])  # (1,1)
    kmeans.partitions[1].add_point(small_point_dataset[1])  # (1,2)
    kmeans.partitions[2].add_point(small_point_dataset[2])  # (1,4)
    kmeans.partitions[3].add_point(small_point_dataset[3])  # (2,1)
    kmeans.partitions[3].add_point(small_point_dataset[4])  # (2,3)
    kmeans.partitions[3].add_point(small_point_dataset[5])  # (3,2)
    kmeans.partitions[3].add_point(small_point_dataset[6])  # (3,4)
    kmeans.partitions[3].add_point(small_point_dataset[7])  # (4,1)
    kmeans.partitions[3].add_point(small_point_dataset[8])  # (4,3)
    kmeans.partitions[3].add_point(small_point_dataset[9])  # (4,4)

    # Update the centroids
    kmeans._update_centroids()

    # Check that the centroids are updated correctly
    assert kmeans.centroids[0].get_x() == pytest.approx(
        1
    ), f"The x coordinate of the first centroid is not correct. The x coordinate of the first centroid is {kmeans.centroids[0].get_x()} and it should be 1."
    assert kmeans.centroids[0].get_y() == pytest.approx(
        1
    ), f"The y coordinate of the first centroid is not correct. The y coordinate of the first centroid is {kmeans.centroids[0].get_y()} and it should be 1."

    assert kmeans.centroids[1].get_x() == pytest.approx(
        1
    ), f"The x coordinate of the second centroid is not correct. The x coordinate of the second centroid is {kmeans.centroids[1].get_x()} and it should be 1."
    assert kmeans.centroids[1].get_y() == pytest.approx(
        2
    ), f"The y coordinate of the second centroid is not correct. The y coordinate of the second centroid is {kmeans.centroids[1].get_y()} and it should be 2."

    assert kmeans.centroids[2].get_x() == pytest.approx(
        1
    ), f"The x coordinate of the third centroid is not correct. The x coordinate of the third centroid is {kmeans.centroids[2].get_x()} and it should be 1."
    assert kmeans.centroids[2].get_y() == pytest.approx(
        4
    ), f"The y coordinate of the third centroid is not correct. The y coordinate of the third centroid is {kmeans.centroids[2].get_y()} and it should be 4."

    assert kmeans.centroids[3].get_x() == pytest.approx(
        3.142857142857143
    ), f"The x coordinate of the fourth centroid is not correct. The x coordinate of the fourth centroid is {kmeans.centroids[3].get_x()} and it should be about 3.142857142857143."
    assert kmeans.centroids[3].get_y() == pytest.approx(
        2.5714285714285716
    ), f"The y coordinate of the fourth centroid is not correct. The y coordinate of the fourth centroid is {kmeans.centroids[3].get_y()} and it should be about 2.5714285714285716."
