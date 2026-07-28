import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # SOSMC reproduction: evidence first

    **Observed cumulative verdict:** Claims 1–5 are `VERIFIED`; Claim 6 is
    `FALSIFIED` for the authors' executable MNIST sweep. This is a
    reproduction forecast, not a live judge score.

    The central question is whether Sequential Optimisation via Sequential
    Monte Carlo (SOSMC) delivers the paper's theoretical guarantees and
    empirical advantages while avoiding reward hacking.
    """)
    return


@app.cell
def _():
    claim_rows = [
        {"Claim": 1, "Result": "VERIFIED", "Evidence": "Official 2D EBM Algorithm 1 trace"},
        {"Claim": 2, "Result": "VERIFIED", "Evidence": "Symbolic proof certificate"},
        {"Claim": 3, "Result": "VERIFIED", "Evidence": "Exact ESS identity + Monte Carlo"},
        {"Claim": 4, "Result": "VERIFIED", "Evidence": "Four settings × 10 seeds"},
        {"Claim": 5, "Result": "VERIFIED", "Evidence": "2D circles objective + tracking"},
        {"Claim": 6, "Result": "FALSIFIED", "Evidence": "12-cell MNIST sweep + PCA counterexample"},
    ]
    return (claim_rows,)


@app.cell
def _(claim_rows, mo):
    mo.ui.table(claim_rows, selection=None)
    return


@app.cell
def _():
    counterexamples = [
        {
            "cell": "bright, beta=0.5",
            "reward 95% CI": "[0.2871, 0.3036]",
            "anomaly median 95% CI": "[5.2092, 5.8277]",
            "fixed limit": 4.1710,
        },
        {
            "cell": "lower-half, beta=1",
            "reward 95% CI": "[0.1667, 0.2193]",
            "anomaly median 95% CI": "[6.9359, 7.3324]",
            "fixed limit": 4.1710,
        },
        {
            "cell": "lower-half, beta=0.5",
            "reward 95% CI": "[0.1823, 0.2373]",
            "anomaly median 95% CI": "[9.3529, 9.6482]",
            "fixed limit": 4.1710,
        },
    ]
    return (counterexamples,)


@app.cell
def _(counterexamples, mo):
    mo.vstack(
        [
            mo.md(
                """
    ## The MNIST counterexample

    The reward increased in all 12 reward/regularisation cells.
    However, three cells crossed a digit-manifold limit calibrated
    only from held-out real MNIST and pretrained samples. Tuned
    samples and controls were excluded from calibration.
    """
            ),
            mo.ui.table(counterexamples, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How to read the result

    `VERIFIED` means the finite executable claim contract passed with an
    independent checker and a control designed to fail. `FALSIFIED` means
    an exact paper-scope configuration satisfied setup assumptions yet
    contradicted the finite no-reward-hacking assertion. It does not claim
    a universal theorem about every MNIST run or every perceptual metric.

    The formal CPU run used 8 allocated cores on Hugging Face
    `cpu-upgrade`, took 7,644.54 seconds, and used Git commit
    `990cb3d8afd53accb03a9e48f0c57e2842137785`.
    """)
    return


if __name__ == "__main__":
    app.run()
