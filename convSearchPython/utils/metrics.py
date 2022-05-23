"""Collection of metrics"""
from typing import List

from convSearchPython.basics import pyterrier as pt
if not pt.started():
    raise Exception('something\'s wrong with pyterrier init')
import pyterrier.measures as measures


def parse_metrics(metrics_str: str) -> List:
    """Parse a comma-separated string of metrics to
    an actual list of metrics"""
    metrics = []
    for m in metrics_str.split(','):
        m = m.strip()
        try:
            if '@' in m:
                m, cutoff = m.split('@', maxsplit=1)
                m = m.strip()
                cutoff = int(cutoff.strip())
                metrics.append(getattr(measures, m) @ cutoff)
            else:
                metrics.append(getattr(measures, m))
        except AttributeError:
            raise Exception(f'invalid metric {m}')
        except ValueError:
            raise Exception(f'invalid cutoff {cutoff}')
    return metrics
