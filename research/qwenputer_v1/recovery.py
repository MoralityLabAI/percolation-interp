"""Exact scoring helpers for a prospective planted-circuit benchmark."""
from fractions import Fraction


def _ids(values, label):
    values = tuple(values)
    if not values or any(type(x) is not str or not x for x in values):
        raise ValueError(f'{label}: nonempty string identifiers required')
    if len(set(values)) != len(values):
        raise ValueError(f'{label}: duplicate identifiers')
    return values


def score_ranking(ranking, universe, planted, budget):
    universe, ranking, planted = (_ids(x, name) for x, name in
                                  ((universe, 'universe'), (ranking, 'ranking'), (planted, 'planted')))
    if set(ranking) != set(universe) or not set(planted) <= set(universe):
        raise ValueError('ranking coverage or answer-key mismatch')
    if type(budget) is not int or not 1 <= budget <= len(universe):
        raise ValueError('invalid matched component budget')
    selected = ranking[:budget]
    hits = len(set(selected) & set(planted))
    return dict(selected=selected, precision=Fraction(hits, budget),
                recall=Fraction(hits, len(planted)), hits=hits, budget=budget)


def paired_summary(candidate, baseline):
    """Descriptive paired differences; this does not issue a statistical verdict."""
    if not candidate or candidate.keys() != baseline.keys():
        raise ValueError('nonempty identical case IDs required')
    values = list(candidate.values()) + list(baseline.values())
    if any(isinstance(v, bool) or not isinstance(v, (int, Fraction)) or not 0 <= v <= 1 for v in values):
        raise ValueError('exact scores in [0,1] required')
    delta = [Fraction(candidate[k]) - Fraction(baseline[k]) for k in candidate]
    return dict(cases=len(delta), mean_difference=sum(delta, Fraction()) / len(delta),
                wins=sum(v > 0 for v in delta), ties=sum(v == 0 for v in delta),
                losses=sum(v < 0 for v in delta), verdict='descriptive_only')


def _projector(p):
    p = tuple(tuple(row) for row in p)
    n = len(p)
    if not n or any(len(row) != n for row in p):
        raise ValueError('square projector required')
    if any(isinstance(x, bool) or not isinstance(x, (int, Fraction)) for row in p for x in row):
        raise ValueError('exact projector required')
    p = tuple(tuple(Fraction(x) for x in row) for row in p)
    if any(p[i][j] != p[j][i] or sum(p[i][k]*p[k][j] for k in range(n)) != p[i][j]
           for i in range(n) for j in range(n)):
        raise ValueError('symmetric idempotent projector required')
    return p


def subspace_recall(recovered, planted):
    """tr(P_recovered P_planted)/rank(P_planted), independent of basis choice."""
    recovered, planted = _projector(recovered), _projector(planted)
    if len(recovered) != len(planted):
        raise ValueError('projector dimensions differ')
    n = len(planted)
    rank = sum(planted[i][i] for i in range(n))
    if not rank:
        raise ValueError('nonzero planted subspace required')
    return sum(recovered[i][j]*planted[j][i] for i in range(n) for j in range(n)) / rank
