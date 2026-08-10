from amdi.pareto import best_under_budgets, pareto_front


def test_pareto_front_rmse():
    rows = [
        {"C_rel": 0.05, "RMSE": 0.10, "status": "ok"},
        {"C_rel": 0.10, "RMSE": 0.08, "status": "ok"},
        {"C_rel": 0.20, "RMSE": 0.09, "status": "ok"},
        {"C_rel": 0.40, "RMSE": 0.06, "status": "ok"},
    ]
    front = pareto_front(rows, "C_rel", "RMSE", minimize_y=True)
    assert [(r["C_rel"], r["RMSE"]) for r in front] == [(0.05, 0.10), (0.10, 0.08), (0.40, 0.06)]


def test_best_under_complexity_budget():
    rows = [
        {"C_rel": 0.05, "RMSE": 0.10, "status": "ok"},
        {"C_rel": 0.08, "RMSE": 0.09, "status": "ok"},
        {"C_rel": 0.20, "RMSE": 0.05, "status": "ok"},
    ]
    selected = best_under_budgets(rows, budgets=(0.10,), metric="RMSE")
    assert len(selected) == 1
    assert selected[0]["C_rel"] == 0.08
