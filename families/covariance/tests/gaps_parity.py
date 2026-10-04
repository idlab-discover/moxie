"""Strict regression gate for repaired audit gaps; not a full parity claim."""
import audit
import shrinkage_boundaries
import graphical_extensions

shrinkage_boundaries.main()
graphical_extensions.main()

# Reuse the pinned-source oracle and fixtures, without widening the reference.
audit.refits()
audit.diagnostics()
failures = []
for item in audit.REPORT['differences']:
    case, method = item['case'], item['method']
    # Reference CD buffer errors remain an explicit extension. Shrinkage
    # numerical failures in both dtypes and invalid-input paths must match.
    if (method == 'diagnostics'
            or case.endswith(('-invalid', '-parameter'))
            or ('-float64-numerical' in case)
            or (case.startswith(('LedoitWolf-', 'OAS-')) and case.endswith('-numerical'))):
        failures.append(item)
assert not failures, failures
print('PASS: pinned sklearn failed-refit state and diagnostic regressions')
