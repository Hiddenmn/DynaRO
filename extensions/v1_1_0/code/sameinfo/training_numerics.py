def accumulate(model, parameters, pair, primary, auxiliary):
    """Exact sequential objective of the successful local resource probe."""
    from regional_budget_objective import LossWeights, sequential_paired_window_backward
    from regional_state_guard import accumulate_current_gradients
    labels, inlet, connected, aa, ab, times, ta, tb = pair
    measures = {}
    for target, weights, names in (
        (primary, LossWeights(site=1., region=1., response=0.), ("site", "region")),
        (auxiliary, LossWeights(site=0., region=0., response=1.), ("response",)),
    ):
        model.zero_grad(set_to_none=True)
        result = sequential_paired_window_backward(
            model, labels, inlet, connected, aa, ab, times, ta, tb,
            start=0, stop=len(aa), weights=weights, gradient_scale=1/4,
            memory_efficient=True)
        accumulate_current_gradients(parameters, target)
        model.zero_grad(set_to_none=True)
        measures.update({name: result[name] for name in names})
    return measures
