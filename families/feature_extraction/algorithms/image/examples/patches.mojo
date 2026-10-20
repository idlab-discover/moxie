"""Extract every overlapping 2-by-2 patch, then reconstruct the image."""
from numojo import zeros, Shape
from moxie.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d, grid_to_graph

def main() raises:
    var image = zeros[DType.float64](Shape(3,4))
    for i in range(image.size):
        image.unsafe_set(i,Float64(i))
    var patches = extract_patches_2d(image,(2,2))
    var restored = reconstruct_from_patches_2d(patches,Shape(3,4))
    var graph = grid_to_graph(3,4,return_as="dense")
    print("patch count:", patches.shape[0])
    print("restored bottom-right:", restored.unsafe_get(11))
    print("graph vertices:", graph.shape[0])
