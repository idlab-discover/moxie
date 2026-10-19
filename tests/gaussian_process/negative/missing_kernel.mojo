from moxie.gaussian_process._plumbing import KernelContract
struct MissingKernel(KernelContract):
    comptime Input = Int
    comptime Params = Int
def main():
    var invalid = MissingKernel()
