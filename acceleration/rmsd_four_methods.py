#!/usr/bin/env python3
"""Run the delivered RMSD CLI methods in one interpreter, retaining four labels.

No RMSD arithmetic is reimplemented. The installed library's own CLI main
function parses fresh inputs for each distinct method; stdout/exception
parsing matches the delivered caller's first-token rule. The duplicate
Hungarian result is reused only if the installed default is Hungarian.
"""
import contextlib
import io
import json
import sys
import traceback


def calculate(opt_xyz, original_xyz):
    from rmsd import calculate_rmsd as rmsd
    base = ['-e', str(opt_xyz), str(original_xyz)]
    settings = rmsd.parse_arguments(base)
    default_is_hungarian = settings.reorder_method == rmsd.REORDER_HUNGARIAN
    values = []
    method_records = []
    plain = None
    for extra, tag in (([], 'plain'), (['--reorder-method', 'hungarian'], 'hungarian'),
                       (['--reorder-method', 'inertia-hungarian'], 'inertia-hungarian'),
                       (['--reorder-method', 'distance'], 'distance')):
        if tag == 'hungarian' and default_is_hungarian:
            record = dict(plain, method=tag, reused_default=True)
        else:
            capture = io.StringIO()
            error = None
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                try:
                    rmsd.main(['-e', *extra, str(opt_xyz), str(original_xyz)])
                except BaseException as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    error = type(exc).__name__
                    traceback.print_exc()
            output = capture.getvalue()
            value = None
            try:
                value = float(output.strip().split()[0])
            except (ValueError, IndexError):
                pass
            record = {'method': tag, 'stdout': output, 'value': value,
                      'error': error, 'reused_default': False}
            if tag == 'plain':
                plain = record
        method_records.append(record)
        if record['value'] is not None:
            values.append((record['value'], tag))
    return {'values': values, 'methods': method_records,
            'default_hungarian_alias_verified': default_is_hungarian}


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: rmsd_four_methods.py OPT_XYZ ORIGINAL_XYZ')
    print(json.dumps(calculate(*sys.argv[1:])))


if __name__ == '__main__':
    main()
