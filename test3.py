from zope.interface import Interface, implementer
from zope import interface

class ISprite(Interface):
    bounding_box = interface.Attribute(
        "The bounding box"
    )

    def intersects(box):
        "Does this intersect with a box"


@implementer(ISprite)
class CircleSprite:
    x: float
    y: float
    radius: float

    @property
    def bounding_box(self):
        return (
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius,
        )

    def intersects(self, box):
        # A box intersects a circle if and only if
        # at least one corner is inside the circle.
        top_left, bottom_right = box[:2], box[2:]
        for choose_x_from in (top_left, bottom_right):
            for choose_y_from in (top_left, bottom_right):
                x = choose_x_from[0]
                y = choose_y_from[1]
                if (((x - self.x) ** 2 + (y - self.y) ** 2) <=
                    self.radius ** 2):
                     return True
        return False