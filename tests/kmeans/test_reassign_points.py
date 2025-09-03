from kmeans import KMeans

from classes.point import Point

#####
# Test(s) with the small dataset of 2D points
#####


def test_with_small_point_dataset_two_partitions_and_reassignments(small_point_dataset):
    """Test the reassignment of points to the partitions with the small dataset of 2D points, 2 partitions and some reassignments."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Initialize the partitions (manually, to be able to only test _reassign_points())
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

    # Update the centroids (manually, to be able to only test _reassign_points())
    kmeans.centroids[0] = Point(1, 2.333333)
    kmeans.centroids[1] = Point(3.142857, 2.571429)

    # Reassign the points to the partitions
    reassign_indicator = kmeans._reassign_points()

    # Check that the points are reassigned correctly
    assert (
        small_point_dataset[0] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[0]} should be in the first partition. The distance to the first centroid is {small_point_dataset[0].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[0].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[1] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[1]} should be in the first partition. The distance to the first centroid is {small_point_dataset[1].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[1].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[2] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[2]} should be in the first partition. The distance to the first centroid is {small_point_dataset[2].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[2].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[3] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[3]} should be in the first partition. The distance to the first centroid is {small_point_dataset[3].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[3].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[4] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[4]} should be in the first partition. The distance to the first centroid is {small_point_dataset[4].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[4].get_distance(kmeans.centroids[1])}."

    assert (
        small_point_dataset[5] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[5]} should be in the second partition. The distance to the first centroid is {small_point_dataset[5].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[5].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[6] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[6]} should be in the second partition. The distance to the first centroid is {small_point_dataset[6].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[6].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[7] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[7]} should be in the second partition. The distance to the first centroid is {small_point_dataset[7].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[7].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[8] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[8]} should be in the second partition. The distance to the first centroid is {small_point_dataset[8].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[8].get_distance(kmeans.centroids[1])}."
    assert (
        small_point_dataset[9] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[9]} should be in the second partition. The distance to the first centroid is {small_point_dataset[9].get_distance(kmeans.centroids[0])} and the distance to the second centroid is {small_point_dataset[9].get_distance(kmeans.centroids[1])}."

    # Check that the reassignment indicator is True (points have been reassigned)
    assert reassign_indicator, "The reassignment indicator should be True."


def test_with_small_point_dataset_three_partitions_and_reassignments(
    small_point_dataset,
):
    """Test the reassignment of points to the partitions with the small dataset of 2D points, 3 partitions and some reassignments."""

    # Create a K-Means instance with 3 clusters
    kmeans = KMeans(3)

    # Initialize the partitions (manually, to be able to only test _reassign_points())
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

    # Update the centroids (manually, to be able to only test _reassign_points())
    kmeans.centroids[0] = Point(1.333333, 1.333333)
    kmeans.centroids[1] = Point(2.25, 3.25)
    kmeans.centroids[2] = Point(4, 2.666667)

    # Reassign the points to the partitions
    reassign_indicator = kmeans._reassign_points()

    # Check that the points are reassigned correctly
    assert (
        small_point_dataset[0] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[0]} should be in the first partition. The distance to the first centroid is {small_point_dataset[0].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[0].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[0].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[1] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[1]} should be in the first partition. The distance to the first centroid is {small_point_dataset[1].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[1].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[1].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[3] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[3]} should be in the first partition. The distance to the first centroid is {small_point_dataset[3].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[3].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[3].get_distance(kmeans.centroids[2])}."

    assert (
        small_point_dataset[2] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[2]} should be in the second partition. The distance to the first centroid is {small_point_dataset[2].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[2].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[2].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[4] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[4]} should be in the second partition. The distance to the first centroid is {small_point_dataset[4].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[4].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[4].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[6] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[6]} should be in the third partition. The distance to the first centroid is {small_point_dataset[6].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[6].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[6].get_distance(kmeans.centroids[2])}."

    assert (
        small_point_dataset[5] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[5]} should be in the second partition. The distance to the first centroid is {small_point_dataset[5].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[5].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[5].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[7] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[7]} should be in the third partition. The distance to the first centroid is {small_point_dataset[7].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[7].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[7].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[8] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[8]} should be in the third partition. The distance to the first centroid is {small_point_dataset[8].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[8].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[8].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[9] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[9]} should be in the third partition. The distance to the first centroid is {small_point_dataset[9].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[9].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[9].get_distance(kmeans.centroids[2])}."

    # Check that the reassignment indicator is True (points have been reassigned)
    assert reassign_indicator, "The reassignment indicator should be True."


def test_with_small_point_dataset_three_partitions_and_no_reassignments(
    small_point_dataset,
):
    """Test the reassignment of points to the partitions with the small dataset of 2D points, 3 partitions and no reassignments."""

    # Create a K-Means instance with 3 clusters
    kmeans = KMeans(3)

    # Initialize the partitions (manually, to be able to only test _reassign_points())
    kmeans.partitions[0].add_point(small_point_dataset[0])  # (1,1)
    kmeans.partitions[0].add_point(small_point_dataset[1])  # (1,2)
    kmeans.partitions[0].add_point(small_point_dataset[3])  # (2,1)
    kmeans.partitions[1].add_point(small_point_dataset[2])  # (1,4)
    kmeans.partitions[1].add_point(small_point_dataset[4])  # (2,3)
    kmeans.partitions[1].add_point(small_point_dataset[6])  # (3,4)
    kmeans.partitions[2].add_point(small_point_dataset[5])  # (3,2)
    kmeans.partitions[2].add_point(small_point_dataset[7])  # (4,1)
    kmeans.partitions[2].add_point(small_point_dataset[8])  # (4,3)
    kmeans.partitions[2].add_point(small_point_dataset[9])  # (4,4)

    # Update the centroids (manually, to be able to only test _reassign_points())
    kmeans.centroids[0] = Point(1.333333, 1.333333)
    kmeans.centroids[1] = Point(2, 3.666667)
    kmeans.centroids[2] = Point(3.75, 2.5)

    # Reassign the points to the partitions
    reassign_indicator = kmeans._reassign_points()

    # Check that the points are still in the correct partitions
    assert (
        small_point_dataset[0] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[0]} should be in the first partition. The distance to the first centroid is {small_point_dataset[0].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[0].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[0].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[1] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[1]} should be in the first partition. The distance to the first centroid is {small_point_dataset[1].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[1].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[1].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[3] in kmeans.partitions[0]
    ), f"The point {small_point_dataset[3]} should be in the first partition. The distance to the first centroid is {small_point_dataset[3].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[3].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[3].get_distance(kmeans.centroids[2])}."

    assert (
        small_point_dataset[2] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[2]} should be in the second partition. The distance to the first centroid is {small_point_dataset[2].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[2].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[2].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[4] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[4]} should be in the second partition. The distance to the first centroid is {small_point_dataset[4].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[4].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[4].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[6] in kmeans.partitions[1]
    ), f"The point {small_point_dataset[6]} should be in the third partition. The distance to the first centroid is {small_point_dataset[6].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[6].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[6].get_distance(kmeans.centroids[2])}."

    assert (
        small_point_dataset[5] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[5]} should be in the second partition. The distance to the first centroid is {small_point_dataset[5].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[5].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[5].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[7] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[7]} should be in the third partition. The distance to the first centroid is {small_point_dataset[7].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[7].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[7].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[8] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[8]} should be in the third partition. The distance to the first centroid is {small_point_dataset[8].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[8].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[8].get_distance(kmeans.centroids[2])}."
    assert (
        small_point_dataset[9] in kmeans.partitions[2]
    ), f"The point {small_point_dataset[9]} should be in the third partition. The distance to the first centroid is {small_point_dataset[9].get_distance(kmeans.centroids[0])}, the distance to the second centroid is {small_point_dataset[9].get_distance(kmeans.centroids[1])}, and the distance to the third centroid is {small_point_dataset[9].get_distance(kmeans.centroids[2])}."

    # Check that the reassignment indicator is False (no points have been reassigned)
    assert not reassign_indicator, "The reassignment indicator should be False."


#####
# Test(s) with the special 2D point dataset to enforce specific edge cases
#####


def test_with_special_points_to_enforce_two_identical_distances_and_point_in_first_partition():
    """Test the reassignment of points to the partitions with a special dataset of 2D points to enforce a identical distance to the centroid of the own partition and the centroid of another partition. The point with the two identical distances is part of the first partition and should not be reassigned to the second partition."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Initialize the partitions (manually, to be able to only test _reassign_points())
    kmeans.partitions[0].add_point(Point(0, 0))
    kmeans.partitions[0].add_point(Point(2, 2))
    kmeans.partitions[1].add_point(Point(3, 3))

    # Update the centroids (manually, to be able to only test _reassign_points())
    kmeans.centroids[0] = Point(1, 1)
    kmeans.centroids[1] = Point(3, 3)

    # Reassign the points to the partitions
    reassign_indicator = kmeans._reassign_points()

    # Check that the points are not reassigned:
    # The distance of (2,2) to both centroids is the same.
    # If the distance of a point to the centroid of its own partition is the smallest distance, the point should not be reassigned to another partition, even if the distance to the centroid of another partition is the same.
    # This is to avoid an infinite loop of reassignments, where the point is reassigned back and forth between the partitions.
    assert (
        Point(0, 0) in kmeans.partitions[0]
    ), f"The point (0,0) should be in the first partition. The distance to the first centroid is {Point(0,0).get_distance(kmeans.centroids[0])} and the distance to the second centroid is {Point(0,0).get_distance(kmeans.centroids[1])}."
    assert (
        Point(2, 2) in kmeans.partitions[0]
    ), f"The point (2,2) should stay in the first partition. While it has the same distance to both centroids, it should not be reassigned to the second partition, to avoid an infinite loop of reassignments."

    assert (
        Point(3, 3) in kmeans.partitions[1]
    ), f"The point (3,3) should be in the second partition. The distance to the first centroid is {Point(3,3).get_distance(kmeans.centroids[0])} and the distance to the second centroid is {Point(3,3).get_distance(kmeans.centroids[1])}."

    # Check that the reassignment indicator is False (no points have been reassigned)
    assert not reassign_indicator, "The reassignment indicator should be False."


def test_with_special_points_to_enforce_two_identical_distances_and_point_in_second_partition():
    """Test the reassignment of points to the partitions with a special dataset of 2D points to enforce a identical distance to the centroid of the own partition and the centroid of another partition. The point with the two identical distances is part of the second partition and should not be reassigned to the first partition."""

    # Create a K-Means instance with 2 clusters
    kmeans = KMeans(2)

    # Initialize the partitions (manually, to be able to only test _reassign_points())
    kmeans.partitions[0].add_point(Point(0, 0))
    kmeans.partitions[1].add_point(Point(1, 1))
    kmeans.partitions[1].add_point(Point(3, 3))

    # Update the centroids (manually, to be able to only test _reassign_points())
    kmeans.centroids[0] = Point(0, 0)
    kmeans.centroids[1] = Point(2, 2)

    # Reassign the points to the partitions
    reassign_indicator = kmeans._reassign_points()

    # Check that the points are not reassigned:
    # The distance of (1,1) to both centroids is the same.
    # If the distance of a point to the centroid of its own partition is the smallest distance, the point should not be reassigned to another partition, even if the distance to the centroid of another partition is the same.
    # This is to avoid an infinite loop of reassignments, where the point is reassigned back and forth between the partitions.
    assert (
        Point(0, 0) in kmeans.partitions[0]
    ), f"The point (0,0) should be in the first partition. The distance to the first centroid is {Point(0,0).get_distance(kmeans.centroids[0])} and the distance to the second centroid is {Point(0,0).get_distance(kmeans.centroids[1])}."

    assert (
        Point(1, 1) in kmeans.partitions[1]
    ), f"The point (1,1) should stay in the second partition. While it has the same distance to both centroids, it should not be reassigned to the first partition, to avoid an infinite loop of reassignments."
    assert (
        Point(3, 3) in kmeans.partitions[1]
    ), f"The point (3,3) should be in the second partition. The distance to the first centroid is {Point(3,3).get_distance(kmeans.centroids[0])} and the distance to the second centroid is {Point(3,3).get_distance(kmeans.centroids[1])}."

    # Check that the reassignment indicator is False (no points have been reassigned)
    assert not reassign_indicator, "The reassignment indicator should be False."


def test_with_special_points_to_enforce_empty_partitions_after_reassignment():
    """Test the reassignment of points to the partitions with a special dataset of 2D points to enforce empty partitions after reassignment. This is an edge case, to check if _reassign_points() handles this case correctly by moving a single point to the empty partition."""

    # Create a K-Means instance with 6 clusters
    kmeans = KMeans(6)

    # Initialize the partitions (manually, to be able to only test _reassign_points())
    kmeans.partitions[0].add_point(Point(3, 6))

    kmeans.partitions[1].add_point(Point(1, 2))
    kmeans.partitions[1].add_point(Point(1, 4))

    kmeans.partitions[2].add_point(Point(0, 3))
    kmeans.partitions[2].add_point(Point(6, 3))

    kmeans.partitions[3].add_point(Point(5, 3))

    kmeans.partitions[4].add_point(Point(3, 0))

    kmeans.partitions[5].add_point(Point(0, 4))
    kmeans.partitions[5].add_point(Point(6, 4))

    # Update the centroids (manually, to be able to only test _reassign_points())
    kmeans.centroids[0] = Point(3, 6)
    kmeans.centroids[1] = Point(1, 3)
    kmeans.centroids[2] = Point(3, 3)
    kmeans.centroids[3] = Point(5, 3)
    kmeans.centroids[4] = Point(3, 0)
    kmeans.centroids[5] = Point(3, 4)

    # Reassign the points to the partitions
    reassign_indicator = kmeans._reassign_points()

    # Outcome of the reassignment would be
    # Partition 0: (3,6)
    # Partition 1: (1,2), (1,4), (0,3), (0,4)
    # Partition 2: EMPTY
    # Partition 3: (5,3), (6,3), (6,4)
    # Partition 4: (3,0)
    # Partition 5: EMPTY
    #
    # Exception handling for Partition 2 and Partition 6 is necessary to avoid empty partitions

    # Make sure that all partitions are non-empty
    for partition_index in range(6):
        assert (
            len(kmeans.partitions[partition_index]) > 0
        ), f"Partition {partition_index} is empty."
