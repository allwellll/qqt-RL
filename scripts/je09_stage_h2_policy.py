"""Fixed long budget; small capability diagnostics never stop training."""
TRAINING_GENERATIONS = 32
POINTS = [0, 4, 8, 12, 16, 24, 32]


def observation(scores):
    return dict(scores=list(scores), may_stop_training=False, observation_only=True)


def hard_safety(row):
    names = row['funnel_names']; groups = row['funnel_by_lesson_family']
    count = lambda name: sum(v[names.index(name)] for scene in groups for v in scene)
    denominator = row['fresh_ticks']//row['generation']
    rates = {k: count(k)/denominator for k in ['death', 'self', 'trade']}
    # Gross native safety failures only; ordinary capability loss is observed.
    stop = rates['death'] >= .02 or rates['self'] >= .005 or rates['trade'] >= .005
    return dict(stop=stop, rates=rates, limits=dict(death=.02, self=.005, trade=.005))
