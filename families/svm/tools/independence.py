"""Reassemble Moxie and exercise public SVMs from the resulting namespace."""
import check

check.toolchain()
check.assemble(check.ROOT)
for source in (
    check.ROOT / 'tests/test_linear_fluent.mojo',
    check.ROOT / 'algorithms/c_svm/tests/test_kernel.mojo',
):
    binary = check.ROOT / 'builds' / ('independence_' + source.stem)
    check.compile_file(source, binary)
    check.run(binary)
print('PASS fresh core assembly and public linear/kernel lifecycle without standalone sources')
