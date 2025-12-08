# Distance from c1 center to c2 center
def circle_distance(c1, c2):
    x1, y1 = c1.center
    x2, y2 = c2.center
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Do c1 and c2 overlap
def circles_overlap(c1, c2):
    dist = circle_distance(c1, c2)
    return dist <= (c1.radius + c2.radius)

# Is c_test inside c_gt
def circle_inside(c_test, c_gt):
    dist = circle_distance(c_test, c_gt)
    return dist + c_test.radius <= c_gt.radius

def check_accuracy(c1, c2):
    """
    Check if one located circle is inside of or touches a ground truth circle.

    Args:
        c1: Detected circle
        c2: Ground truth circle

    Returns:
        True if c1 is inside or touches c2, False otherwise
    """

    if circle_inside(c1, c2):
        return True

    if circles_overlap(c1, c2):
        return True

    return False
