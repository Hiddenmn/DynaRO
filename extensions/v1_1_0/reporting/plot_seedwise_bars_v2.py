from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
DEV_SUMMARY=HERE/'inputs/learning_dev.json'
TEST_SUMMARY=HERE/'inputs/learning_test.json'
SEEDS=(20260811,20260812,20260813)

def load_rows(dev_path=DEV_SUMMARY, test_path=TEST_SUMMARY) -> list[dict[str, float | int | str]]:
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    if dev["status"] != "COMPLETE_STATEGUARD_THREE_SEED_ADAPTIVE_DEV_AGGREGATION":
        raise ValueError("DEV source is not the completed three-seed aggregate")
    if test["status"] != "COMPLETE_FROZEN_LOCKED_TEST" or test["pair_count"] != 600:
        raise ValueError("Locked TEST source is not the completed 600-pair evaluation")

    rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        dev_scores = dev["per_seed"][str(seed)]["equal_block_mean"]
        test_scores = {
            arm: test["per_checkpoint"][f"{arm}/{seed}"]["overall_equal_block_mean"]["all_25"]
            for arm in ("response50", "stateguard")
        }
        for split, pair_count, scores in (
            ("Development", 400, dev_scores),
            ("Locked test", 600, test_scores),
        ):
            fixed, state = scores["response50"], scores["stateguard"]
            if fixed["site"] <= 0 or fixed["response"] <= 0:
                raise ValueError("Comparator errors must be positive for relative reductions")
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "pairs": pair_count,
                    "fixed50_site": fixed["site"],
                    "stateguard_site": state["site"],
                    "fixed50_response": fixed["response"],
                    "stateguard_response": state["response"],
                    "site_reduction_pct": 100 * (fixed["site"] - state["site"]) / fixed["site"],
                    "response_reduction_pct": 100 * (fixed["response"] - state["response"]) / fixed["response"],
                }
            )
    return rows
