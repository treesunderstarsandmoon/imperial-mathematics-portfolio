class Circle:

    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius
    
    def __contains__(self,other):
        r1 = self.centre[0] - other[0]
        r2 = self.centre[1] - other[1]
        if r1 ** 2 + r2 ** 2 < self.radius ** 2:
            return True
        else:
            return False

