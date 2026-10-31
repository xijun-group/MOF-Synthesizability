"""Portable, dependency-free definition of the 3.1.1 Ehull stopping protocol."""
import math
import os

PROTOCOL_ID = "mofsynth-ehull-combined-v1"
DEFAULTS = dict(protocol_id=PROTOCOL_ID, atom_fmax_eV_A=0.01, filter_fmax=0.03,
                stress_GPa=0.05, energy_window_steps=5, energy_span_eV_atom=1e-5,
                consecutive_steps=5)


def protocol(**overrides):
    result = dict(DEFAULTS)
    if set(overrides) - set(result):
        raise ValueError("Unknown convergence parameter")
    result.update(overrides)
    if result['protocol_id'] != PROTOCOL_ID:
        raise ValueError("Unsupported Ehull convergence protocol")
    for key in ('atom_fmax_eV_A', 'filter_fmax', 'stress_GPa', 'energy_span_eV_atom'):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
        result[key] = value
    for key in ('energy_window_steps', 'consecutive_steps'):
        value = result[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f'{key} must be a positive integer')
        result[key] = int(value)
    return result


def from_environment():
    names = {'atom_fmax_eV_A': ('EHULL_ATOM_FMAX', float),
             'filter_fmax': ('EHULL_FMAX', float),
             'stress_GPa': ('EHULL_STRESS_MAX_GPA', float),
             'energy_window_steps': ('EHULL_ENERGY_WINDOW_STEPS', int),
             'energy_span_eV_atom': ('EHULL_ENERGY_SPAN_PER_ATOM', float),
             'consecutive_steps': ('EHULL_CONSECUTIVE_STEPS', int)}
    return protocol(**{key: cast(os.environ[name]) for key, (name, cast) in names.items()
                       if os.environ.get(name)})


def evaluate(history, settings):
    row = history[-1]
    window = history[-(settings['energy_window_steps'] + 1):]
    energies = [r['energy_per_atom_eV'] for r in window]
    span = max(energies) - min(energies) if len(window) > settings['energy_window_steps'] else None
    gates = dict(atom_force=row['atom_fmax_eV_A'] < settings['atom_fmax_eV_A'],
                 generalized_force=row['filter_fmax'] < settings['filter_fmax'],
                 stress=row['stress_max_abs_principal_GPa'] < settings['stress_GPa'],
                 energy=span is not None and span <= settings['energy_span_eV_atom'])
    return gates, span


def matches_receipt(receipt, expected):
    final = receipt.get('final', {})
    gates = final.get('gates', {})
    return (receipt.get('convergence_protocol') == expected
            and receipt.get('converged') is True
            and receipt.get('termination_reason') == 'combined_converged'
            and final.get('consecutive_passes', 0) >= expected['consecutive_steps']
            and all(gates.get(k) is True for k in ('atom_force', 'generalized_force', 'stress', 'energy')))
