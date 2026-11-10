"""Compare the LLM judge with the hand labels, and chart where the agent fails.

Run: python -m scripts.judge_agreement
Reads the hand-labelled file and an eval run of the same 48 questions, prints a summary
(judge agreement, TPR/TNR, code checks) and saves PNG charts in results/all-plots/plots_<run>/.
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt

LABELS_PATH = "results/labels_2026-11-09.json"
RUN_PATH = "results/golden_2026-11-10_rejudge_v2.jsonl"  # the run to compare with the hand labels
OUT_DIR = Path("results/all-plots") / ("plots_" + Path(RUN_PATH).stem.split("_", 2)[2])  # e.g. plots_rejudge_v2
TOP_CAUSES = 4  # show this many causes, group the rest as "Other"
COST_CEILING = 0.005  # dollars per question; 4 of 48 answers in final_48 cost more
NUMBER_IN_TEXT = re.compile(r"(\d[\d,]*\.\d+)\s*%?|(\d[\d,]*)\s*%")  # decimals like 2.72, or percentages like 21%

# Colours (validated categorical palette, light mode): blue = correct, orange = wrong
CORRECT = "#2a78d6"
WRONG = "#eb6834"
NEUTRAL = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def load():
    """Join each hand label with its saved eval row (same order, checked by question text)."""
    with open(LABELS_PATH, encoding="utf-8") as file:
        labels = json.load(file)
    with open(RUN_PATH, encoding="utf-8") as file:
        runs = [json.loads(line) for line in file]
    items = []
    for label, run in zip(labels, runs, strict=True):
        assert label["question"] == run["question"], f"#{label['n']}: labels and run are out of order"
        assert label["label"] is not None, f"#{label['n']} has no hand label yet"
        evidence, found = label["evidence_pages"], label["evidence_retrieved"]
        items.append({
            "n": label["n"],
            "company": label["company"],
            "type": run["question_type"],
            "human": label["label"],
            "judge": run["correct"],
            "cause": label["cause"].strip() or "(none given)",
            "short": "short answer" in label["note"].lower(),
            "evidence": "all" if set(evidence) <= set(found) else ("some" if found else "none"),
            "confidence": run["confidence"],
            "forced": any("last step" in step for step in run.get("steps", [])),
            "calculator": bool(run.get("calculations")),
            "cost": run["cost_usd"] or 0,
            "answer_chars": len(run["answer"]),
            "unbacked": unbacked_numbers(run),
            "checks": {
                "figure check clean": not run["warnings"],
                "no fallback": not run["fallback"],
                f"cost under ${COST_CEILING}": (run["cost_usd"] or 0) <= COST_CEILING,
                "finished within MAX_STEPS": not any("last step" in step for step in run.get("steps", [])),
                "text numbers backed": not unbacked_numbers(run),
            },
        })
    return items


def unbacked_numbers(run: dict) -> list[str]:
    """Numbers in the answer text that match no listed figure or calculator result.

    Only decimals and percentages are checked (years and plain counts are too common).
    Numbers the question itself contains (e.g. '20%' in 'more than 20%') are skipped.
    """
    values = [f["value"] for f in run["figures"]] + [c["result"] for c in run["calculations"]]
    unbacked = []
    for match in NUMBER_IN_TEXT.finditer(run["answer"]):
        text = match.group(1) or match.group(2)
        if text in run["question"]:
            continue
        number = float(text.replace(",", ""))
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9  # '2.72' matches anything from 2.715 to 2.725
        # a figure may be stored as a fraction (0.1652 -> 16.52%), in millions ($1.5 billion), or negative
        candidates = [abs(v * scale) for v in values for scale in (1, 100, 1 / 1000)]
        if not any(abs(c - number) <= tolerance for c in candidates):
            unbacked.append(text)
    return unbacked


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved {OUT_DIR / name}")


def label_bars(ax, bars, horizontal=False, fmt="{:.0f}"):
    """Write each bar's value just past its end (skips zeros)."""
    for bar in bars:
        value = bar.get_width() if horizontal else bar.get_height()
        if not value:
            continue
        if horizontal:
            ax.text(bar.get_x() + value + 0.15, bar.get_y() + bar.get_height() / 2, fmt.format(value),
                    va="center", color=INK_2, fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, fmt.format(value),
                    ha="center", va="bottom", color=INK_2, fontsize=9)


def stacked_by(items, key, title, name, order=None):
    """Horizontal stacked bars: hand-labelled correct vs wrong for each value of `key`."""
    groups = order or sorted({i[key] for i in items}, key=lambda g: -sum(i[key] == g for i in items))
    right = [sum(i[key] == g and i["human"] for i in items) for g in groups]
    wrong = [sum(i[key] == g and not i["human"] for i in items) for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(groups) + 1.4))
    y = range(len(groups))
    ax.barh(y, right, color=CORRECT, height=0.6, label="correct", edgecolor=SURFACE, linewidth=2)
    ax.barh(y, wrong, left=right, color=WRONG, height=0.6, label="wrong", edgecolor=SURFACE, linewidth=2)
    for row, (r, w) in enumerate(zip(right, wrong)):
        share = f"{r}/{r + w} ({r / (r + w):.0%})" if r + w else ""
        ax.text(r + w + 0.15, row, share, va="center", color=INK_2, fontsize=9)
    ax.set_yticks(list(y), [str(g) for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r + w for r, w in zip(right, wrong)) * 1.3)
    ax.set_xlabel("questions (hand labels)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    save(fig, name)


def main():
    items = load()
    total = len(items)
    tp = sum(i["human"] and i["judge"] for i in items)          # both say correct
    fn = sum(i["human"] and not i["judge"] for i in items)      # judge too strict
    fp = sum(not i["human"] and i["judge"] for i in items)      # judge too lenient
    tn = sum(not i["human"] and not i["judge"] for i in items)  # both say wrong
    tpr = tp / (tp + fn) if tp + fn else 0  # of the truly correct answers, share the judge accepts
    tnr = tn / (tn + fp) if tn + fp else 0  # of the truly wrong answers, share the judge rejects
    human_score, judge_score = tp + fn, tp + fp

    print(f"\nHand labels: {human_score}/{total} correct   Judge: {judge_score}/{total} correct")
    print(f"Agreement: {tp + tn}/{total} ({(tp + tn) / total:.0%})")
    print(f"TPR {tpr:.1%} ({tp}/{tp + fn})   TNR {tnr:.1%} ({tn}/{tn + fp})")
    disagree = [i for i in items if i["human"] != i["judge"]]
    print("Disagreements (#n: human/judge): "
          + ", ".join(f"#{i['n']} {'T' if i['human'] else 'F'}/{'T' if i['judge'] else 'F'}" for i in disagree))
    print("\nSaving charts:")

    # 1. Confusion matrix
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cells = [[tp, fn], [fp, tn]]
    ax.imshow([[1, 0], [0, 1]], cmap=matplotlib.colors.ListedColormap(["#f0efec", "#cde2fb"]))
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cells[r][c]), ha="center", va="center", fontsize=22, fontweight="bold", color=INK)
    ax.set_xticks([0, 1], ["judge: correct", "judge: wrong"])
    ax.set_yticks([0, 1], ["you: correct", "you: wrong"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Judge vs hand labels ({tp + tn}/{total} agree)")
    save(fig, "01_confusion_matrix.png")

    # 2. TPR and TNR
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(["TPR\naccepts correct answers", "TNR\nrejects wrong answers"], [tpr * 100, tnr * 100],
                  color=[CORRECT, WRONG], width=0.5)
    for bar, text in zip(bars, [f"{tpr:.0%} ({tp}/{tp + fn})", f"{tnr:.0%} ({tn}/{tn + fp})"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, text, ha="center", color=INK_2)
    ax.set_ylim(0, 115)
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.set_title("How trustworthy is the judge?")
    save(fig, "02_tpr_tnr.png")

    # 3. Headline score: hand labels vs judge
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bars = ax.barh(["judge", "hand labels"], [judge_score, human_score], color=[NEUTRAL, CORRECT], height=0.5)
    for bar in bars:
        value = int(bar.get_width())
        ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2, f"{value}/{total} ({value / total:.0%})",
                va="center", color=INK_2)
    ax.set_xlim(0, total)
    ax.set_xlabel("correct answers")
    ax.grid(axis="y", visible=False)
    ax.set_title("Score on the 48-question golden set")
    save(fig, "03_score_human_vs_judge.png")

    # 4. Causes of wrong answers (top few, the rest grouped as Other)
    causes = Counter(i["cause"] for i in items if not i["human"]).most_common()
    top, rest = causes[:TOP_CAUSES], causes[TOP_CAUSES:]
    if rest:
        top.append(("Other: " + ", ".join(c for c, _ in rest), sum(k for _, k in rest)))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(top) + 1.3))
    bars = ax.barh([c for c, _ in top], [k for _, k in top], color=WRONG, height=0.6)
    label_bars(ax, bars, horizontal=True)
    ax.invert_yaxis()
    ax.set_xlabel("wrong answers")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Why the agent was wrong ({total - human_score} wrong answers)")
    save(fig, "04_failure_causes.png")

    # 5. Why the judge disagreed with you: tags on the answers it rejected but you accepted
    strict = [i for i in items if i["human"] and not i["judge"]]
    reasons = Counter("reference questionable" if i["cause"] == "reference questionable"
                      else "short answer" if i["short"] else "other (wording, detail, rounding)"
                      for i in strict).most_common()
    if reasons:
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(reasons) + 1.3))
        bars = ax.barh([r for r, _ in reasons], [k for _, k in reasons], color=NEUTRAL, height=0.6)
        label_bars(ax, bars, horizontal=True)
        ax.invert_yaxis()
        ax.set_xlabel("answers")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Correct answers the judge rejected ({len(strict)}), by tag")
        save(fig, "05_judge_too_strict.png")

    # 6-11. Correct vs wrong (hand labels) split by one property at a time
    stacked_by(items, "company", "Correct vs wrong by company", "06_by_company.png")
    stacked_by(items, "type", "Correct vs wrong by question type", "07_by_question_type.png")
    stacked_by(items, "evidence", "Were the evidence pages retrieved?", "08_by_evidence_retrieved.png",
               order=["all", "some", "none"])
    stacked_by(items, "confidence", "Does the agent's confidence predict correctness?", "09_by_confidence.png",
               order=["high", "low"])
    for i in items:
        i["forced_text"] = "forced at last step" if i["forced"] else "finished on its own"
        i["calc_text"] = "used the calculator" if i["calculator"] else "no calculator"
        i["short_text"] = "tagged short answer" if i["short"] else "not tagged short"
    stacked_by(items, "forced_text", "Forced answers vs finished ones", "10_by_forced.png")
    stacked_by(items, "calc_text", "Calculator use", "11_by_calculator.png")
    stacked_by(items, "short_text", "Short answers (tag in your notes)", "12_by_short_answer.png")

    # 13. Judge vs hand score per company (where does the judge's strictness land?)
    companies = sorted({i["company"] for i in items})
    human_c = [sum(i["company"] == c and i["human"] for i in items) for c in companies]
    judge_c = [sum(i["company"] == c and i["judge"] for i in items) for c in companies]
    sizes = [sum(i["company"] == c for i in items) for c in companies]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(companies) + 1.4))
    y = range(len(companies))
    ax.barh([v - 0.18 for v in y], human_c, height=0.34, color=CORRECT, label="hand labels")
    ax.barh([v + 0.18 for v in y], judge_c, height=0.34, color=NEUTRAL, label="judge")
    for row, size in enumerate(sizes):
        ax.text(max(human_c[row], judge_c[row]) + 0.15, row, f"of {size}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(list(y), companies)
    ax.invert_yaxis()
    ax.set_xlabel("correct answers")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Correct answers per company: hand labels vs judge")
    save(fig, "13_company_human_vs_judge.png")

    # 14. Answer length: correct vs wrong (one dot per question)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for row, (flag, colour, name) in enumerate([(True, CORRECT, "correct"), (False, WRONG, "wrong")]):
        xs = [i["answer_chars"] for i in items if i["human"] == flag]
        ax.scatter(xs, [row] * len(xs), s=40, color=colour, alpha=0.8, edgecolors=SURFACE, linewidths=1.5)
        if xs:
            median = sorted(xs)[len(xs) // 2]
            ax.plot([median, median], [row - 0.3, row + 0.3], color=INK, linewidth=2)
            ax.text(median, row + 0.36, f"median {median}", ha="center", color=INK_2, fontsize=9)
    ax.set_yticks([0, 1], ["correct", "wrong"])
    ax.set_ylim(-0.6, 1.8)
    ax.invert_yaxis()
    ax.set_xlabel("answer length (characters)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Answer length, correct vs wrong")
    save(fig, "14_answer_length.png")

    # 15. Code checks: pass rate, and how often answers are right when a check passes vs fails
    print("\nCode checks (accuracy uses the hand labels):")
    print(f"  {'check':<28}{'pass':>8}{'right when pass':>18}{'right when fail':>18}")
    names = list(items[0]["checks"])
    pass_rate, right_pass, right_fail = [], [], []
    for name in names:
        passed = [i for i in items if i["checks"][name]]
        failed = [i for i in items if not i["checks"][name]]
        rp = sum(i["human"] for i in passed) / len(passed) if passed else 0
        rf = sum(i["human"] for i in failed) / len(failed) if failed else None
        pass_rate.append(len(passed) / total)
        right_pass.append(rp)
        right_fail.append(rf)
        fail_text = "-" if rf is None else f"{rf:.0%} of {len(failed)}"
        print(f"  {name:<28}{len(passed):>5}/{total}{rp:>12.0%} of {len(passed):<4}{fail_text:>14}")
    flagged = [i for i in items if i["unbacked"]]
    print("  Unbacked numbers: " + ", ".join(f"#{i['n']} {i['unbacked']}" for i in flagged))

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.5))
    y = range(len(names))
    ax.barh([v - 0.18 for v in y], [r * 100 for r in right_pass], height=0.34, color=CORRECT,
            label="answers right when the check passes")
    ax.barh([v + 0.18 for v in y], [0 if r is None else r * 100 for r in right_fail], height=0.34,
            color=WRONG, label="answers right when the check fails")
    for row, (rp, rf) in enumerate(zip(right_pass, right_fail)):
        ax.text(rp * 100 + 1, row - 0.18, f"{rp:.0%}", va="center", color=INK_2, fontsize=9)
        ax.text((rf or 0) * 100 + 1, row + 0.18, "never fails" if rf is None else f"{rf:.0%}",
                va="center", color=INK_2, fontsize=9)
    ax.set_yticks(list(y), [f"{n}\n(passes {p:.0%})" for n, p in zip(names, pass_rate)])
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of answers correct (hand labels)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Do the code checks catch wrong answers?", pad=28)
    save(fig, "15_code_checks.png")

    print(f"\nDone: {len(list(OUT_DIR.glob('*.png')))} charts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
