"""Checked image geometry, separate from tabular estimator metadata."""
from numojo import Shape


def checked_extent_product(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0:
        raise Error("ValueError: negative extent")
    if b != 0 and a > Int.MAX // b:
        raise Error("ValueError: shape product overflows Int")
    return a * b


struct PatchGeometry(Copyable, Movable):
    var height: Int
    var width: Int
    var channels: Int
    var patch_height: Int
    var patch_width: Int

    def __init__(out self, height: Int, width: Int, patch_height: Int,
                 patch_width: Int, channels: Int = 1) raises:
        if height <= 0 or width <= 0 or channels <= 0:
            raise Error("ValueError: image dimensions must be positive")
        if patch_height <= 0 or patch_width <= 0 or patch_height > height or patch_width > width:
            raise Error("ValueError: patch dimensions must fit the image")
        _ = checked_extent_product(checked_extent_product(height, width), channels)
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_height = patch_height
        self.patch_width = patch_width

    def positions(self) raises -> Int:
        return checked_extent_product(self.height - self.patch_height + 1,
                                      self.width - self.patch_width + 1)

    def output_shape(self, count: Int) raises -> Shape:
        _ = checked_extent_product(count, checked_extent_product(
            checked_extent_product(self.patch_height, self.patch_width), self.channels))
        if self.channels == 1:
            return Shape(count, self.patch_height, self.patch_width)
        return Shape(count, self.patch_height, self.patch_width, self.channels)


struct GridGeometry(Copyable, Movable):
    var n_x: Int
    var n_y: Int
    var n_z: Int

    def __init__(out self, n_x: Int, n_y: Int, n_z: Int = 1) raises:
        _ = checked_extent_product(checked_extent_product(n_x, n_y), n_z)
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

    def vertices(self) raises -> Int:
        return checked_extent_product(checked_extent_product(self.n_x, self.n_y), self.n_z)

    def dense_shape(self, selected: Int) raises -> Shape:
        if selected < 0 or selected > self.vertices():
            raise Error("ValueError: invalid selected vertex count")
        _ = checked_extent_product(selected, selected)
        return Shape(selected, selected)
