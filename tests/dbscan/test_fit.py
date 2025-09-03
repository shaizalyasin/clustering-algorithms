from dbscan import DBSCAN

from classes.point import Point

#####
# Test(s) with the small dataset of 2D points
#####


def test_with_small_point_dataset_epsilon_1_and_min_pts_2(small_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the small dataset of 2D points, epsilon=1, and min_pts=2.
    """

    # Create a DBSCAN instance with epsilon=1 and min_pts=2
    dbscan = DBSCAN(1, 2)

    # Fit the DBSCAN instance to the small dataset
    dbscan.fit(small_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 2
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 2. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 4
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 4. (Noisy points: {dbscan.noise})"

    # Check that the points are clustered correctly
    # Get the cluster that contains the point (1,1)
    clusters_with_1_1 = [
        cluster for cluster in dbscan.clusters if Point(1, 1) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (1,1)
    assert len(clusters_with_1_1) > 0, "There is no cluster that contains (1,1)."
    assert (
        len(clusters_with_1_1) < 2
    ), "There is more than one cluster that contains (1,1)."

    # Check that the cluster that contains the point (1,1) contains the correct points
    assert (
        len(clusters_with_1_1[0]) == 3
    ), f"The cluster that contains (1,1) does not contain the correct number of points. The cluster contains {len(clusters_with_1_1[0])} points and it should contain 3 points."
    assert (
        Point(1, 1) in clusters_with_1_1[0]
    ), "The point (1,1) is not in the cluster that contains (1,1)."
    assert (
        Point(1, 2) in clusters_with_1_1[0]
    ), "The point (1,2) is not in the cluster that contains (1,1)."
    assert (
        Point(2, 1) in clusters_with_1_1[0]
    ), "The point (2,1) is not in the cluster that contains (1,1)."

    # Get the cluster that contains the point (4,4)
    clusters_with_4_4 = [
        cluster for cluster in dbscan.clusters if Point(4, 4) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (4,4)
    assert len(clusters_with_4_4) > 0, "There is no cluster that contains (4,4)."
    assert (
        len(clusters_with_4_4) < 2
    ), "There is more than one cluster that contains (4,4)."

    # Check that the cluster that contains the point (4,4) contains the correct points
    assert (
        len(clusters_with_4_4[0]) == 3
    ), f"The cluster that contains (4,4) does not contain the correct number of points. The cluster contains {len(clusters_with_4_4[0])} points and it should contain 3 points."
    assert (
        Point(3, 4) in clusters_with_4_4[0]
    ), "The point (3,4) is not in the cluster that contains (4,4)."
    assert (
        Point(4, 3) in clusters_with_4_4[0]
    ), "The point (4,3) is not in the cluster that contains (4,4)."
    assert (
        Point(4, 4) in clusters_with_4_4[0]
    ), "The point (4,4) is not in the cluster that contains (4,4)."

    # Check that the noise points are correct
    assert (
        Point(1, 4) in dbscan.noise
    ), "The point (1,4) is not correctly identified as noise."
    assert (
        Point(2, 3) in dbscan.noise
    ), "The point (2,3) is not correctly identified as noise."
    assert (
        Point(3, 2) in dbscan.noise
    ), "The point (3,2) is not correctly identified as noise."
    assert (
        Point(4, 1) in dbscan.noise
    ), "The point (4,1) is not correctly identified as noise."


def test_with_small_point_dataset_epsilon_1_and_min_pts_3(small_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the small dataset of 2D points, epsilon=1, and min_pts=3.
    """

    # Create a DBSCAN instance with epsilon=1 and min_pts=3
    dbscan = DBSCAN(1, 3)

    # Fit the DBSCAN instance to the small dataset
    dbscan.fit(small_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 2
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 2. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 4
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 4. (Noisy points: {dbscan.noise})"

    # Check that the points are clustered correctly
    # Get the cluster that contains the point (1,1)
    clusters_with_1_1 = [
        cluster for cluster in dbscan.clusters if Point(1, 1) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (1,1)
    assert len(clusters_with_1_1) > 0, "There is no cluster that contains (1,1)."
    assert (
        len(clusters_with_1_1) < 2
    ), "There is more than one cluster that contains (1,1)."

    # Check that the cluster that contains the point (1,1) contains the correct points
    assert (
        len(clusters_with_1_1[0]) == 3
    ), f"The cluster that contains (1,1) does not contain the correct number of points. The cluster contains {len(clusters_with_1_1[0])} points and it should contain 3 points."
    assert (
        Point(1, 1) in clusters_with_1_1[0]
    ), "The point (1,1) is not in the cluster that contains (1,1)."
    assert (
        Point(1, 2) in clusters_with_1_1[0]
    ), "The point (1,2) is not in the cluster that contains (1,1)."
    assert (
        Point(2, 1) in clusters_with_1_1[0]
    ), "The point (2,1) is not in the cluster that contains (1,1)."

    # Get the cluster that contains the point (4,4)
    clusters_with_4_4 = [
        cluster for cluster in dbscan.clusters if Point(4, 4) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (4,4)
    assert len(clusters_with_4_4) > 0, "There is no cluster that contains (4,4)."
    assert (
        len(clusters_with_4_4) < 2
    ), "There is more than one cluster that contains (4,4)."

    # Check that the cluster that contains the point (4,4) contains the correct points
    assert (
        len(clusters_with_4_4[0]) == 3
    ), f"The cluster that contains (4,4) does not contain the correct number of points. The cluster contains {len(clusters_with_4_4[0])} points and it should contain 3 points."
    assert (
        Point(3, 4) in clusters_with_4_4[0]
    ), "The point (3,4) is not in the cluster that contains (4,4)."
    assert (
        Point(4, 3) in clusters_with_4_4[0]
    ), "The point (4,3) is not in the cluster that contains (4,4)."
    assert (
        Point(4, 4) in clusters_with_4_4[0]
    ), "The point (4,4) is not in the cluster that contains (4,4)."

    # Check that the noise points are correct
    assert (
        Point(1, 4) in dbscan.noise
    ), "The point (1,4) is not correctly identified as noise."
    assert (
        Point(2, 3) in dbscan.noise
    ), "The point (2,3) is not correctly identified as noise."
    assert (
        Point(3, 2) in dbscan.noise
    ), "The point (3,2) is not correctly identified as noise."
    assert (
        Point(4, 1) in dbscan.noise
    ), "The point (4,1) is not correctly identified as noise."


def test_with_small_point_dataset_epsilon_1_414214_and_min_pts_3(small_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the small dataset of 2D points, epsilon=1.414214, and min_pts=3 (Result should be one big cluster containing all points).
    """

    # Create a DBSCAN instance with epsilon=1.414214 and min_pts=3
    dbscan = DBSCAN(1.414214, 3)

    # Fit the DBSCAN instance to the small dataset
    dbscan.fit(small_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 1
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 1. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 0
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 0. (Noisy points: {dbscan.noise})"

    # Check that the points are clustered correctly
    # Get the cluster that contains the point (1,1)
    clusters_with_1_1 = [
        cluster for cluster in dbscan.clusters if Point(1, 1) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (1,1)
    assert len(clusters_with_1_1) > 0, "There is no cluster that contains (1,1)."
    assert (
        len(clusters_with_1_1) < 2
    ), "There is more than one cluster that contains (1,1)."

    # Check that the cluster that contains the point (1,1) contains the correct points
    assert (
        len(clusters_with_1_1[0]) == 10
    ), f"The cluster that contains (1,1) does not contain the correct number of points. The cluster contains {len(clusters_with_1_1[0])} points and it should contain 10 points."
    assert (
        Point(1, 1) in clusters_with_1_1[0]
    ), "The point (1,1) is not in the cluster that contains (1,1)."
    assert (
        Point(1, 2) in clusters_with_1_1[0]
    ), "The point (1,2) is not in the cluster that contains (1,1)."
    assert (
        Point(1, 4) in clusters_with_1_1[0]
    ), "The point (1,4) is not in the cluster that contains (1,1)."
    assert (
        Point(2, 1) in clusters_with_1_1[0]
    ), "The point (2,1) is not in the cluster that contains (1,1)."
    assert (
        Point(2, 3) in clusters_with_1_1[0]
    ), "The point (2,3) is not in the cluster that contains (1,1)."
    assert (
        Point(3, 2) in clusters_with_1_1[0]
    ), "The point (3,2) is not in the cluster that contains (1,1)."
    assert (
        Point(3, 4) in clusters_with_1_1[0]
    ), "The point (3,4) is not in the cluster that contains (1,1)."
    assert (
        Point(4, 1) in clusters_with_1_1[0]
    ), "The point (4,1) is not in the cluster that contains (1,1)."
    assert (
        Point(4, 3) in clusters_with_1_1[0]
    ), "The point (4,3) is not in the cluster that contains (1,1)."
    assert (
        Point(4, 4) in clusters_with_1_1[0]
    ), "The point (4,4) is not in the cluster that contains (1,1)."


def test_with_small_point_dataset_epsilon_3_and_min_pts_10(small_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the small dataset of 2D points, epsilon=3, and min_pts=10 (Result should be one big cluster containing all points).
    """

    # Create a DBSCAN instance with epsilon=3 and min_pts=10
    dbscan = DBSCAN(3, 10)

    # Fit the DBSCAN instance to the small dataset
    dbscan.fit(small_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 1
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 1. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 0
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 0. (Noisy points: {dbscan.noise})"

    # Check that the points are clustered correctly
    # Get the cluster that contains the point (1,1)
    clusters_with_1_1 = [
        cluster for cluster in dbscan.clusters if Point(1, 1) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (1,1)
    assert len(clusters_with_1_1) > 0, "There is no cluster that contains (1,1)."
    assert (
        len(clusters_with_1_1) < 2
    ), "There is more than one cluster that contains (1,1)."

    # Check that the cluster that contains the point (1,1) contains the correct points
    assert (
        len(clusters_with_1_1[0]) == 10
    ), f"The cluster that contains (1,1) does not contain the correct number of points. The cluster contains {len(clusters_with_1_1[0])} points and it should contain 10 points."
    assert (
        Point(1, 1) in clusters_with_1_1[0]
    ), "The point (1,1) is not in the cluster that contains (1,1)."
    assert (
        Point(1, 2) in clusters_with_1_1[0]
    ), "The point (1,2) is not in the cluster that contains (1,1)."
    assert (
        Point(1, 4) in clusters_with_1_1[0]
    ), "The point (1,4) is not in the cluster that contains (1,1)."
    assert (
        Point(2, 1) in clusters_with_1_1[0]
    ), "The point (2,1) is not in the cluster that contains (1,1)."
    assert (
        Point(2, 3) in clusters_with_1_1[0]
    ), "The point (2,3) is not in the cluster that contains (1,1)."
    assert (
        Point(3, 2) in clusters_with_1_1[0]
    ), "The point (3,2) is not in the cluster that contains (1,1)."
    assert (
        Point(3, 4) in clusters_with_1_1[0]
    ), "The point (3,4) is not in the cluster that contains (1,1)."
    assert (
        Point(4, 1) in clusters_with_1_1[0]
    ), "The point (4,1) is not in the cluster that contains (1,1)."
    assert (
        Point(4, 3) in clusters_with_1_1[0]
    ), "The point (4,3) is not in the cluster that contains (1,1)."
    assert (
        Point(4, 4) in clusters_with_1_1[0]
    ), "The point (4,4) is not in the cluster that contains (1,1)."


def test_with_small_point_dataset_epsilon_1_and_min_pts_4(small_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the small dataset of 2D points, epsilon=1, and min_pts=4 (Result should be that all points are noise).
    """

    # Create a DBSCAN instance with epsilon=1 and min_pts=4
    dbscan = DBSCAN(1, 4)

    # Fit the DBSCAN instance to the small dataset
    dbscan.fit(small_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 0
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 0. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 10
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 10. (Noisy points: {dbscan.noise})"

    # Check that the noise points are correct
    assert (
        Point(1, 1) in dbscan.noise
    ), "The point (1,1) is not correctly identified as noise."
    assert (
        Point(1, 2) in dbscan.noise
    ), "The point (1,2) is not correctly identified as noise."
    assert (
        Point(1, 4) in dbscan.noise
    ), "The point (1,4) is not correctly identified as noise."
    assert (
        Point(2, 1) in dbscan.noise
    ), "The point (2,1) is not correctly identified as noise."
    assert (
        Point(2, 3) in dbscan.noise
    ), "The point (2,3) is not correctly identified as noise."
    assert (
        Point(3, 2) in dbscan.noise
    ), "The point (3,2) is not correctly identified as noise."
    assert (
        Point(3, 4) in dbscan.noise
    ), "The point (3,4) is not correctly identified as noise."
    assert (
        Point(4, 1) in dbscan.noise
    ), "The point (4,1) is not correctly identified as noise."
    assert (
        Point(4, 3) in dbscan.noise
    ), "The point (4,3) is not correctly identified as noise."
    assert (
        Point(4, 4) in dbscan.noise
    ), "The point (4,4) is not correctly identified as noise."


def test_with_small_point_dataset_epsilon_2_5_and_min_pts_10(small_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the small dataset of 2D points, epsilon=2.5, and min_pts=10 (Result should be that all points are noise).
    """

    # Create a DBSCAN instance with epsilon=2.5 and min_pts=10
    dbscan = DBSCAN(2.5, 10)

    # Fit the DBSCAN instance to the small dataset
    dbscan.fit(small_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 0
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 0. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 10
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 10. (Noisy points: {dbscan.noise})"

    # Check that the noise points are correct
    assert (
        Point(1, 1) in dbscan.noise
    ), "The point (1,1) is not correctly identified as noise."
    assert (
        Point(1, 2) in dbscan.noise
    ), "The point (1,2) is not correctly identified as noise."
    assert (
        Point(1, 4) in dbscan.noise
    ), "The point (1,4) is not correctly identified as noise."
    assert (
        Point(2, 1) in dbscan.noise
    ), "The point (2,1) is not correctly identified as noise."
    assert (
        Point(2, 3) in dbscan.noise
    ), "The point (2,3) is not correctly identified as noise."
    assert (
        Point(3, 2) in dbscan.noise
    ), "The point (3,2) is not correctly identified as noise."
    assert (
        Point(3, 4) in dbscan.noise
    ), "The point (3,4) is not correctly identified as noise."
    assert (
        Point(4, 1) in dbscan.noise
    ), "The point (4,1) is not correctly identified as noise."
    assert (
        Point(4, 3) in dbscan.noise
    ), "The point (4,3) is not correctly identified as noise."
    assert (
        Point(4, 4) in dbscan.noise
    ), "The point (4,4) is not correctly identified as noise."


#####
# Test(s) with the bigger dataset of 2D points
#####


def test_with_bigger_point_dataset_epsilon_1_and_min_pts_2(bigger_point_dataset):
    """
    Test fitting the DBSCAN clustering instance with the bigger dataset of 2D points, epsilon=1, and min_pts=2.
    """

    # Create a DBSCAN instance with epsilon=1 and min_pts=2
    dbscan = DBSCAN(1, 2)

    # Fit the DBSCAN instance to the bigger dataset
    dbscan.fit(bigger_point_dataset)

    # Check that the number of clusters is correct
    assert (
        len(dbscan.clusters) == 2
    ), f"The number of clusters is not correct. The number of clusters is {len(dbscan.clusters)} and it should be 2. (Clusters: {dbscan.clusters})"

    # Check that the number of points that are identified as noise is correct
    assert (
        len(dbscan.noise) == 38
    ), f"The number of points that are identified as noise is not correct. The number of noise points is {len(dbscan.noise)} and it should be 38. (Noisy points: {dbscan.noise})"

    # Check that the points are clustered correctly
    # Get the cluster that contains the point (3,3)
    clusters_with_3_3 = [
        cluster for cluster in dbscan.clusters if Point(3, 3) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (3,3)
    assert len(clusters_with_3_3) > 0, "There is no cluster that contains (3,3)."
    assert (
        len(clusters_with_3_3) < 2
    ), "There is more than one cluster that contains (3,3)."

    # Check that the cluster that contains the point (3,3) contains the correct amount of points
    assert (
        len(clusters_with_3_3[0]) == 56
    ), f"The cluster that contains (3,3) does not contain the correct number of points. The cluster contains {len(clusters_with_3_3[0])} points and it should contain 56 points."

    # Check that the cluster that contains the point (3,3) contains (5,5) twice
    assert (
        clusters_with_3_3[0].points.count(Point(5, 5)) == 2
    ), f"The point (5,5) should be in the cluster that contains (3,3) twice, but it is in the cluster {clusters_with_3_3[0].points.count(Point(5, 5))} times."

    # Check that all other points that belong to the cluster are present
    points = [
        (3, 4),
        (4, 4),
        (4, 5),
        (5, 6),
        (6, 6),
        (6, 7),
        (7, 7),
        (7, 8),
        (8, 8),
        (8, 9),
        (9, 9),
        (9, 10),
        (10, 10),
        (2, 3),
        (2, 2),
        (1, 2),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 2),
        (-2, 2),
        (-2, 1),
        (-1, 0),
        (0, 0),
        (0, -1),
        (-1, -1),
        (-1, -2),
        (-2, -2),
        (-2, -3),
        (-3, -3),
        (-3, -4),
        (-4, -4),
        (-4, -5),
        (-5, -5),
        (-5, -6),
        (-6, -6),
        (-6, -7),
        (-7, -7),
        (-7, -8),
        (-8, -8),
        (-8, -9),
        (-9, -9),
        (-9, -10),
        (-10, -10),
        (-10, -9),
        (-9, -8),
        (-8, -7),
        (-7, -6),
        (-6, -5),
        (-5, -4),
        (-4, -3),
        (-3, -2),
        (-2, -1),
    ]
    for point in points:
        # Create a Point object
        point = Point(point[0], point[1])

        # Check that the point is in the cluster
        assert (
            point in clusters_with_3_3[0]
        ), f"The point {point} is not in the cluster that contains (3,3)."

    # Get the cluster that contains the point (9,-10)
    clusters_with_9_minus_10 = [
        cluster for cluster in dbscan.clusters if Point(9, -10) in cluster
    ]

    # Check that there is exactly one cluster that contains the point (9,-10)
    assert (
        len(clusters_with_9_minus_10) > 0
    ), "There is no cluster that contains (9,-10)."
    assert (
        len(clusters_with_9_minus_10) < 2
    ), "There is more than one cluster that contains (9,-10)."

    # Check that the cluster that contains the point (9,-10) contains the correct amount of points
    assert (
        len(clusters_with_9_minus_10[0]) == 6
    ), f"The cluster that contains (9,-10) does not contain the correct number of points. The cluster contains {len(clusters_with_9_minus_10[0])} points and it should contain 6 points."

    # Check that all other points that belong to the cluster are present
    points = [(9, -10), (9, -9), (8, -9), (8, -10), (7, -10), (10, -10)]
    for point in points:
        # Create a Point object
        point = Point(point[0], point[1])

        # Check that the point is in the cluster
        assert (
            point in clusters_with_9_minus_10[0]
        ), f"The point {point} is not in the cluster that contains (9,-10)."

    # Check that the noise points are correctly identified
    points = [
        (-9, 9),
        (-8, 8),
        (7, 10),
        (4, 7),
        (2, -5),
        (6, -9),
        (5, -8),
        (-4, 4),
        (-6, 6),
        (3, -4),
        (6, 9),
        (3, -6),
        (-10, 10),
        (-5, -2),
        (1, 4),
        (-7, 7),
        (-10, -7),
        (-4, -1),
        (-3, 3),
        (2, -3),
        (6, -7),
        (0, 3),
        (7, -8),
        (10, -8),
        (4, -7),
        (1, -4),
        (1, -2),
        (-6, -3),
        (3, 6),
        (-3, 0),
        (-9, -6),
        (5, -6),
        (4, -5),
        (-7, -4),
        (-8, -5),
        (5, 8),
        (2, 5),
        (0, -3),
    ]
    for point in points:
        # Create a Point object
        point = Point(point[0], point[1])

        # Check that the point is in the noise
        assert (
            point in dbscan.noise
        ), f"The point {point} is not correctly identified as noise."
