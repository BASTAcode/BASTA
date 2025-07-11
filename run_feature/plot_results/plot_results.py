import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PostProcessingValidator:
    def __init__(self, run_configs, glob_mass_file):

        # columns in BASTA results.ascii output (setting: define_output["outparams"] = ("Teff", "FeH", "logg", "massfin", "age"))
        self.cols = [
            "starid",
            "Teff",
            "Teff_errp",
            "Teff_errm",
            "FeH",
            "FeH_errp",
            "FeH_errm",
            "logg",
            "logg_errp",
            "logg_errm",
            "massfin",
            "massfin_errp",
            "massfin_errm",
            "age",
            "age_errp",
            "age_errm",
        ]

        # load globular cluster literature masses
        self.glob_m = pd.read_csv(
            glob_mass_file,
            sep=r"\s+",
            comment="#",
            names=[
                "cluster",
                "starid",
                "lit_M",
                "lit_M_rand_err",
                "lit_M_sys_err",
                "ref",
            ],
        )
        self.glob_m["starid"] = self.glob_m["starid"].astype(int)

        self.runs = []

        # loop through runs
        for label, run_dir in run_configs:
            result_file = os.path.join(run_dir, "results.ascii")
            df = pd.read_csv(result_file, sep=r"\s+", comment="#", names=self.cols)
            df["starid"] = df["starid"].astype(int)

            # read chi2 from JSON files
            chi2_vals = []
            for sid in df["starid"]:
                json_path = os.path.join(run_dir, f"{sid}.json")
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        data = json.load(f)
                    try:
                        chi2_val = list(data.values())[0]["chi2"][0]
                    except (KeyError, IndexError):
                        chi2_val = np.nan
                else:
                    chi2_val = np.nan
                chi2_vals.append(chi2_val)

            df["chi2"] = chi2_vals
            df.dropna(subset=["chi2"], inplace=True)

            # rescale chi2 for marker size
            df["chi2_scaled"] = 10 + 90 * (df["chi2"] - df["chi2"].min()) / (
                df["chi2"].max() - df["chi2"].min()
            )

            # join with literature masses and compute residuals
            merged_df = pd.merge(df, self.glob_m, on="starid", how="inner")
            self._add_mass_error(merged_df)

            self.runs.append(
                {
                    "label": label,
                    "df": merged_df.copy(),
                }
            )

    def _add_mass_error(self, df):
        # compute total uncertainty and residual in mass
        df["howell_mass_err"] = np.sqrt(
            df["lit_M_rand_err"] ** 2 + df["lit_M_sys_err"] ** 2
        )
        df["mass_diff"] = df["massfin"] - df["lit_M"]
        df["mass_diff_abs"] = df["mass_diff"].abs()

    def plot_residuals(
        self, separate=False, chi2_lim=None, xlim=None, ylim_mass=None, ylim_age=None
    ):
        """
        Plot mass and age residuals.
        - If `separate=True`, creates separate panels for each run (rather than all runs on the same scatter plot)
        - chi2_lim: exclude stars with chi2 > threshold
        - xlim, ylim_mass, ylim_age: set axis limits
        """
        cluster_colors = ["#ef8a62", "#b2182b", "#2166ac", "#67a9cf"]

        if not separate:
            fig, axes = plt.subplots(2, 1, figsize=(10 / 3, 1.5 * 10 / 3), sharey=False)

            for run, color in zip(self.runs, cluster_colors):
                df = run["df"]
                label = run["label"]
                if chi2_lim is not None:
                    label += f" (chi2 < {chi2_lim})"
                    df = df[df["chi2"] <= chi2_lim]

                self._plot_scatters(axes, df, label, color, xlim, ylim_mass, ylim_age)

            axes[0].set_ylabel(r"$|\Delta M|$ [$\mathrm{M}_\odot$]")
            axes[1].set_ylabel("Age [Gyr]")
            axes[1].set_xlabel("Star Index")

            axes[1].axhspan(
                13.8, axes[1].get_ylim()[1], color="gray", alpha=0.3, label="> 13.8 Gyr"
            )
            axes[1].axhspan(
                11, 13, color="lightblue", alpha=0.3, label="rough M4 literature range"
            )

            axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=1)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3)
            plt.show()

        else:
            # one plot per run
            for run, color in zip(self.runs, cluster_colors):
                df = run["df"]
                label = run["label"]
                if chi2_lim is not None:
                    label += f" (chi2 < {chi2_lim})"
                    df = df[df["chi2"] <= chi2_lim]

                fig, axes = plt.subplots(
                    2, 1, figsize=(10 / 3, 1.5 * 10 / 3), sharey=False
                )

                self._plot_scatters(axes, df, label, color, xlim, ylim_mass, ylim_age)

                axes[0].set_ylabel(r"$|\Delta M|$ [$\mathrm{M}_\odot$]")
                axes[1].set_ylabel("Age [Gyr]")
                axes[1].set_xlabel("Star Index")

                axes[1].axhspan(
                    13.8,
                    axes[1].get_ylim()[1],
                    color="gray",
                    alpha=0.3,
                    label="> 13.8 Gyr",
                )
                axes[1].axhspan(
                    11, 13, color="lightblue", alpha=0.3, label="M4 literature range"
                )

                fig.suptitle(label, fontsize="medium")
                axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=1)
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.3)
                plt.show()

    def _plot_scatters(
        self, axes, df, label, color, xlim=None, ylim_mass=None, ylim_age=None
    ):
        # helper function for plot_residuals
        xvals = np.arange(len(df))

        axes[0].scatter(xvals, df["mass_diff_abs"], color=color, s=df["chi2_scaled"])
        axes[1].scatter(
            xvals, df["age"] / 1e3, label=label, color=color, s=df["chi2_scaled"]
        )

        if xlim is not None:
            axes[0].set_xlim(xlim)
            axes[1].set_xlim(xlim)
        if ylim_mass is not None:
            axes[0].set_ylim(ylim_mass)
        if ylim_age is not None:
            axes[1].set_ylim(ylim_age)
