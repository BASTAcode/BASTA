import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PostProcessingValidator:
    def __init__(self, run_configs, open_mass_file, glob_mass_file):

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
        for label, result_file, chi2_file, cluster_type in run_configs:

            df = pd.read_csv(result_file, sep=r"\s+", comment="#", names=self.cols)
            chi2_df = pd.read_csv(
                chi2_file, sep="\t", names=["starid", "chi2"], header=0
            )

            df["starid"] = df["starid"].astype(int)
            chi2_df["starid"] = chi2_df["starid"].astype(int)

            df = pd.merge(df, chi2_df, on="starid", how="left")

            df["chi2_scaled"] = 10 + 90 * (df["chi2"] - df["chi2"].min()) / (
                df["chi2"].max() - df["chi2"].min()
            )

        if cluster_type == "open":
            merged_df = pd.merge(df, self.open_m, on="starid")
        elif cluster_type == "globular":
            merged_df = pd.merge(df, self.glob_m, on="starid")
        else:
            raise ValueError(f"Unknown cluster_type: {cluster_type}")

        self._add_mass_error(merged_df, cluster_type)
        self.runs.append(
            {"label": label, "df": merged_df, "cluster_type": cluster_type}
        )

    def _add_mass_error(self, df, cluster_type):
        if cluster_type == "open":
            df["mass_diff"] = df["massfin"] - df["M"]
            df["mass_diff_abs"] = df["mass_diff"].abs()
            df["massfin_errp"] = df["M_upper_err"]
            df["massfin_errm"] = df["M_lower_err"]
        elif cluster_type == "globular":
            df["howell_mass_err"] = np.sqrt(
                df["lit_M_rand_err"] ** 2 + df["lit_M_sys_err"] ** 2
            )
            df["mass_diff"] = df["massfin"] - df["lit_M"]
            df["mass_diff_abs"] = df["mass_diff"].abs()

    def plot_residuals(self):
        fig, axes = plt.subplots(2, 1, figsize=(10 / 3, 1.5 * 10 / 3), sharey=False)
        cluster_colors = ["#ef8a62", "#b2182b", "#2166ac", "#67a9cf"]

        for run, color in zip(self.runs, cluster_colors):
            df = run["df"]
            label = run["label"]
            xvals = np.arange(len(df))

            axes[0].scatter(
                xvals,
                df["mass_diff_abs"],
                label=label,
                color=color,
                s=df["chi2_scaled"],
            )
            axes[1].scatter(
                xvals, df["age"] / 1e3, label=label, color=color, s=df["chi2_scaled"]
            )

        axes[1].axhspan(
            13.8, axes[1].get_ylim()[1], color="gray", alpha=0.3, label="> 13.8 Gyr"
        )

        axes[1].axhspan(
            11, 13, color="lightblue", alpha=0.3, label="M4 literature range"
        )

        axes[0].set_ylabel(r"$|\Delta M|$ [$\mathrm{M}_\odot$]")
        axes[1].set_ylabel("Age [Gyr]")
        axes[1].set_xlabel("Star Index")

        axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")

        plt.tight_layout()
        plt.subplots_adjust(right=0.78)
        plt.show()
