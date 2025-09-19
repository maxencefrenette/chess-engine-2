# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import wandb

    api = wandb.Api()

    runs = api.runs(path="maxence-frenette/chess-engine-2", filters={"tags": "sota"})

    def extract_flops_and_loss(run):
        flops = run.config.get("flops_budget")
        loss = run.history(keys=["loss"], pandas=True)["loss"].iloc[-100:].mean()
        return {
            "flops": flops,
            "loss": loss
        }

    runs = [extract_flops_and_loss(run) for run in runs]
    runs = pd.DataFrame(runs)
    runs
    return mo, runs


@app.cell
def _(mo, runs):
    import altair as alt

    chart = alt.Chart(runs) \
        .mark_point() \
        .encode(
            x=alt.X("flops", scale=alt.Scale(type="log"), title="FLOPs"),
            y=alt.Y("loss", scale=alt.Scale(type="log"), title="Loss"),
            tooltip=["flops", "loss"]
        ) \
        .properties(
            title="Loss vs FLOPs",
            width=600,
            height=400
        ) \
        .interactive()

    mo.ui.altair_chart(chart)
    return


if __name__ == "__main__":
    app.run()
