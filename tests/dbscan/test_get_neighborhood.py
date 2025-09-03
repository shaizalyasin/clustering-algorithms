from dbscan import DBSCAN

from classes.point import Point

#####
# Test(s) with the small dataset of 2D points
#####


def test_with_small_point_dataset_and_epsilon_1(small_point_dataset):
    """Test the correct identification of neighborhoods with the small dataset of 2D points and epsilon=1"""

    # Create a DBSCAN instance with epsilon=1 and min_points=3 (min_points is not relevant for this test)
    dbscan = DBSCAN(1, 3)

    # Identify all points in the neighborhood of (1,1)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[0], small_point_dataset)

    # Check that the neighborhood contains the points (1,1), (1,2), (2,1) (and only these points)
    assert (
        small_point_dataset[0] in neighborhood
    ), "The point (1,1) is missing in the neighborhood of (1,1)."
    assert (
        small_point_dataset[1] in neighborhood
    ), "The point (1,2) is missing in the neighborhood of (1,1)."
    assert (
        small_point_dataset[3] in neighborhood
    ), "The point (2,1) is missing in the neighborhood of (1,1)."
    assert (
        len(neighborhood) == 3
    ), "The neighborhood of (1,1) contains more points than expected."

    # Identify all points in the neighborhood of (1,2)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[1], small_point_dataset)

    # Check that the neighborhood contains the points (1,1), (1,2) (and only these points)
    assert (
        small_point_dataset[0] in neighborhood
    ), "The point (1,1) is missing in the neighborhood of (1,2)."
    assert (
        small_point_dataset[1] in neighborhood
    ), "The point (1,2) is missing in the neighborhood of (1,2)."
    assert (
        len(neighborhood) == 2
    ), "The neighborhood of (1,2) contains more points than expected."

    # Identify all points in the neighborhood of (1,4)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[2], small_point_dataset)

    # Check that the neighborhood contains only the point (1,4)
    assert (
        small_point_dataset[2] in neighborhood
    ), "The point (1,4) is missing in the neighborhood of (1,4)."
    assert (
        len(neighborhood) == 1
    ), "The neighborhood of (1,4) contains more points than expected."

    # Identify all points in the neighborhood of (2,1)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[3], small_point_dataset)

    # Check that the neighborhood contains the points (1,1), (2,1) (and only these points)
    assert (
        small_point_dataset[0] in neighborhood
    ), "The point (1,1) is missing in the neighborhood of (2,1)."
    assert (
        small_point_dataset[3] in neighborhood
    ), "The point (2,1) is missing in the neighborhood of (2,1)."
    assert (
        len(neighborhood) == 2
    ), "The neighborhood of (2,1) contains more points than expected."

    # Identify all points in the neighborhood of (2,3)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[4], small_point_dataset)

    # Check that the neighborhood contains only the point (2,3)
    assert (
        small_point_dataset[4] in neighborhood
    ), "The point (2,3) is missing in the neighborhood of (2,3)."
    assert (
        len(neighborhood) == 1
    ), "The neighborhood of (2,3) contains more points than expected."

    # Identify all points in the neighborhood of (3,2)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[5], small_point_dataset)

    # Check that the neighborhood contains only the point (3,2)
    assert (
        small_point_dataset[5] in neighborhood
    ), "The point (3,2) is missing in the neighborhood of (3,2)."
    assert (
        len(neighborhood) == 1
    ), "The neighborhood of (3,2) contains more points than expected."

    # Identify all points in the neighborhood of (3,4)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[6], small_point_dataset)

    # Check that the neighborhood contains the points (3,4), (4,4) (and only these points)
    assert (
        small_point_dataset[6] in neighborhood
    ), "The point (3,4) is missing in the neighborhood of (3,4)."
    assert (
        small_point_dataset[9] in neighborhood
    ), "The point (4,4) is missing in the neighborhood of (3,4)."

    # Identify all points in the neighborhood of (4,1)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[7], small_point_dataset)

    # Check that the neighborhood contains only the point (4,1)
    assert (
        small_point_dataset[7] in neighborhood
    ), "The point (4,1) is missing in the neighborhood of (4,1)."
    assert (
        len(neighborhood) == 1
    ), "The neighborhood of (4,1) contains more points than expected."

    # Identify all points in the neighborhood of (4,3)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[8], small_point_dataset)

    # Check that the neighborhood contains the points (4,3), (4,4) (and only these points)
    assert (
        small_point_dataset[8] in neighborhood
    ), "The point (4,3) is missing in the neighborhood of (4,3)."
    assert (
        small_point_dataset[9] in neighborhood
    ), "The point (4,4) is missing in the neighborhood of (4,3)."
    assert (
        len(neighborhood) == 2
    ), "The neighborhood of (4,3) contains more points than expected."

    # Identify all points in the neighborhood of (4,4)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[9], small_point_dataset)

    # Check that the neighborhood contains the points (3,4), (4,3), (4,4) (and only these points)
    assert (
        small_point_dataset[6] in neighborhood
    ), "The point (3,4) is missing in the neighborhood of (4,4)."
    assert (
        small_point_dataset[8] in neighborhood
    ), "The point (4,3) is missing in the neighborhood of (4,4)."
    assert (
        small_point_dataset[9] in neighborhood
    ), "The point (4,4) is missing in the neighborhood of (4,4)."
    assert (
        len(neighborhood) == 3
    ), "The neighborhood of (4,4) contains more points than expected."


def test_with_small_point_dataset_and_epsilon_1_414214(small_point_dataset):
    """Test the correct identification of neighborhoods with the small dataset of 2D points and epsilon=1.414214"""

    # Create a DBSCAN instance with epsilon=1.414214 and min_points=3 (min_points is not relevant for this test)
    dbscan = DBSCAN(1.414214, 3)

    # Identify all points in the neighborhood of (1,1)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[0], small_point_dataset)

    # Check that the neighborhood contains the points (1,1), (1,2), (2,1) (and only these points)
    assert (
        small_point_dataset[0] in neighborhood
    ), "The point (1,1) is missing in the neighborhood of (1,1)."
    assert (
        small_point_dataset[1] in neighborhood
    ), "The point (1,2) is missing in the neighborhood of (1,1)."
    assert (
        small_point_dataset[3] in neighborhood
    ), "The point (2,1) is missing in the neighborhood of (1,1)."
    assert (
        len(neighborhood) == 3
    ), "The neighborhood of (1,1) contains more points than expected."

    # Identify all points in the neighborhood of (1,2)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[1], small_point_dataset)

    # Check that the neighborhood contains the points (1,1), (1,2), (2,1), (2,3) (and only these points)
    assert (
        small_point_dataset[0] in neighborhood
    ), "The point (1,1) is missing in the neighborhood of (1,2)."
    assert (
        small_point_dataset[1] in neighborhood
    ), "The point (1,2) is missing in the neighborhood of (1,2)."
    assert (
        small_point_dataset[3] in neighborhood
    ), "The point (2,1) is missing in the neighborhood of (1,2)."
    assert (
        small_point_dataset[4] in neighborhood
    ), "The point (2,3) is missing in the neighborhood of (1,2)."
    assert (
        len(neighborhood) == 4
    ), "The neighborhood of (1,2) contains more points than expected."

    # Identify all points in the neighborhood of (1,4)
    neighborhood = dbscan._get_neighborhood(small_point_dataset[2], small_point_dataset)

    # Check that the neighborhood contains the points (1,4), (2,3) (and only these points)
    assert (
        small_point_dataset[2] in neighborhood
    ), "The point (1,4) is missing in the neighborhood of (1,4)."
    assert (
        small_point_dataset[4] in neighborhood
    ), "The point (2,3) is missing in the neighborhood of (1,4)."
    assert (
        len(neighborhood) == 2
    ), "The neighborhood of (1,4) contains more points than expected."


#####
# Test(s) with the bigger dataset of 2D points
#####


def test_with_bigger_point_dataset_and_epsilon_3(bigger_point_dataset):
    """Test the correct identification of neighborhoods with the bigger dataset of 2D points and epsilon=3"""

    # Create a DBSCAN instance with epsilon=3 and min_points=2 (min_points is not relevant for this test)
    dbscan = DBSCAN(3, 2)

    # Identify all points in the neighborhood of (-10,-10)
    neighborhood = dbscan._get_neighborhood(Point(-10, -10), bigger_point_dataset)

    # Check that the neighborhood contains the points (-10, -10), (-9, -9), (-8, -8),
    # (-10, -9), (-9, -8), (-9, -10), (-8, -9), (-10, -7) (and only these points)
    assert (
        Point(-10, -10) in neighborhood
    ), "The point (-10, -10) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-9, -9) in neighborhood
    ), "The point (-9, -9) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-8, -8) in neighborhood
    ), "The point (-8, -8) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-10, -9) in neighborhood
    ), "The point (-10, -9) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-9, -8) in neighborhood
    ), "The point (-9, -8) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-9, -10) in neighborhood
    ), "The point (-9, -10) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-8, -9) in neighborhood
    ), "The point (-8, -9) is missing in the neighborhood of (-10, -10)."
    assert (
        Point(-10, -7) in neighborhood
    ), "The point (-10, -7) is missing in the neighborhood of (-10, -10)."
    assert (
        len(neighborhood) == 8
    ), "The neighborhood of (-10, -10) contains more points than expected."

    # Identify all points in the neighborhood of (0,0)
    neighborhood = dbscan._get_neighborhood(Point(0, 0), bigger_point_dataset)

    # Check that the neighborhood contains the points (-2, -2), (-1, -1), (0, 0), (1, 1),
    # (2, 2), (-2, 2), (-1, 1), (0, -1), (1, -2), (-2, -1), (-1, 0), (0, 1), (1, 2),
    # (-1, -2), (0, -3), (-3, 0), (-2, 1), (-1, 2), (0, 3) (and only these points)
    assert (
        Point(-2, -2) in neighborhood
    ), "The point (-2, -2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-1, -1) in neighborhood
    ), "The point (-1, -1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(0, 0) in neighborhood
    ), "The point (0, 0) is missing in the neighborhood of (0, 0)."
    assert (
        Point(1, 1) in neighborhood
    ), "The point (1, 1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(2, 2) in neighborhood
    ), "The point (2, 2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-2, 2) in neighborhood
    ), "The point (-2, 2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-1, 1) in neighborhood
    ), "The point (-1, 1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(0, -1) in neighborhood
    ), "The point (0, -1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(1, -2) in neighborhood
    ), "The point (1, -2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-2, -1) in neighborhood
    ), "The point (-2, -1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-1, 0) in neighborhood
    ), "The point (-1, 0) is missing in the neighborhood of (0, 0)."
    assert (
        Point(0, 1) in neighborhood
    ), "The point (0, 1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(1, 2) in neighborhood
    ), "The point (1, 2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-1, -2) in neighborhood
    ), "The point (-1, -2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(0, -3) in neighborhood
    ), "The point (0, -3) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-3, 0) in neighborhood
    ), "The point (-3, 0) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-2, 1) in neighborhood
    ), "The point (-2, 1) is missing in the neighborhood of (0, 0)."
    assert (
        Point(-1, 2) in neighborhood
    ), "The point (-1, 2) is missing in the neighborhood of (0, 0)."
    assert (
        Point(0, 3) in neighborhood
    ), "The point (0, 3) is missing in the neighborhood of (0, 0)."
    assert (
        len(neighborhood) == 19
    ), "The neighborhood of (0, 0) contains more points than expected."

    # Identify all points in the neighborhood of (5,5)
    neighborhood = dbscan._get_neighborhood(Point(5, 5), bigger_point_dataset)

    # Check that the neighborhood contains the points (3, 3), (4, 4), (6, 6), (7, 7), (3, 4), (4, 5), (5, 6), (6, 7), (2, 5), (3, 6), (4, 7), (5, 8) plus twice (5,5) (and only these points)
    assert (
        Point(3, 3) in neighborhood
    ), "The point (3, 3) is missing in the neighborhood of (5, 5)."
    assert (
        Point(4, 4) in neighborhood
    ), "The point (4, 4) is missing in the neighborhood of (5, 5)."
    assert (
        Point(6, 6) in neighborhood
    ), "The point (6, 6) is missing in the neighborhood of (5, 5)."
    assert (
        Point(7, 7) in neighborhood
    ), "The point (7, 7) is missing in the neighborhood of (5, 5)."
    assert (
        Point(3, 4) in neighborhood
    ), "The point (3, 4) is missing in the neighborhood of (5, 5)."
    assert (
        Point(4, 5) in neighborhood
    ), "The point (4, 5) is missing in the neighborhood of (5, 5)."
    assert (
        Point(5, 6) in neighborhood
    ), "The point (5, 6) is missing in the neighborhood of (5, 5)."
    assert (
        Point(6, 7) in neighborhood
    ), "The point (6, 7) is missing in the neighborhood of (5, 5)."
    assert (
        Point(2, 5) in neighborhood
    ), "The point (2, 5) is missing in the neighborhood of (5, 5)."
    assert (
        Point(3, 6) in neighborhood
    ), "The point (3, 6) is missing in the neighborhood of (5, 5)."
    assert (
        Point(4, 7) in neighborhood
    ), "The point (4, 7) is missing in the neighborhood of (5, 5)."
    assert (
        Point(5, 8) in neighborhood
    ), "The point (5, 8) is missing in the neighborhood of (5, 5)."
    # Check that (5,5) is in the neighborhood twice
    assert (
        neighborhood.count(Point(5, 5)) == 2
    ), f"The point (5, 5) should be twice in the neighborhood of (5, 5), but is only {neighborhood.count(Point(5, 5))} times."
    assert (
        len(neighborhood) == 14
    ), "The neighborhood of (5, 5) contains more points than expected."
