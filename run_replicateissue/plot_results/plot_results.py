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

        # Load maddy's GC masses
        self.glob_m = pd.read_csv(
            glob_mass_file,
            sep=r"\s+",
            comment="#",
            names=[
                "cluster",
                "starid",
                "howell_mass",
                "M_rand_err",
                "M_sys_err",
                "ref",
            ],
        )
        self.glob_m["starid"] = self.glob_m["starid"].astype(int)

        # Load runs
        self.runs = []
        for label, filename, cluster_type in run_configs:
            if cluster_type == "open":
                df = pd.read_csv(filename, sep=r"\s+", comment="#", names=self.cols)
                merged_df = pd.merge(df, self.open_m, on="starid")
            elif cluster_type == "globular":
                df = pd.read_csv(filename, sep=r"\s+", comment="#", names=self.cols)
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
                df["M_rand_err"] ** 2 + df["M_sys_err"] ** 2
            )
            df["mass_diff"] = df["massfin"] - df["howell_mass"]
            df["mass_diff_abs"] = df["mass_diff"].abs()

    def plot_residuals(self):

        for run in self.runs:
            df = run["df"]
            # print(df["mass_diff_abs"])

        fig, axes = plt.subplots(2, 1, figsize=(10 / 3, 1.5 * 10 / 3), sharey=False)

        axes[0].plot(np.arange(0, len(df), 1), df["mass_diff_abs"], "o")
        axes[1].plot(np.arange(0, len(df), 1), (df["age"] * 1e6) / 1e9, "o")

        plt.show()

        # for run, color in zip(self.runs, cluster_colors):
        #     df = run["df"]
        #     label = run["label"]
        #     cluster_type = run["cluster_type"]

        #     if cluster_type == 'open':
        #         obs_mass = df["M"]
        #         obs_err = [df["M_lower_err"], df["M_upper_err"]]
        #         basta_err = [df["massfin_errm"], df["massfin_errp"]]
        #     elif cluster_type == 'globular':
        #         obs_mass = df["howell_mass"]
        #         obs_err = [df["howell_mass_err"], df["howell_mass_err"]]
        #         basta_err = [df["massfin_errm"], df["massfin_errp"]]
        #     else:
        #         continue

        #     ax1.errorbar(
        #         obs_mass, df["massfin"],
        #         xerr=obs_err, yerr=basta_err,
        #         fmt='o', color=color, markeredgecolor='black', label=label, capsize=3
        #     )

        #     feh = df["FeH"]
        #     feh_err = [df["FeH_errm"], df["FeH_errp"]]
        #     age = df["age"] / 1e3
        #     age_err = [df["age_errm"], df["age_errp"]]

        #     ax2.scatter(
        #         age, feh, marker='D', color=color, edgecolor='black', label=label, zorder=2
        #     )


#     def plot_mass_comparison(self):
#         fig, axes = plt.subplots(1, 2, figsize=(2*10/3, 1.9*10/3), sharey=False)

#         ax1, ax2 = axes
#         x = np.linspace(0.4, 2.2, 100)

#         # LEFT: Mass comparison
#         ax1.plot(x, x, 'k--', label="1:1 line")

#         sigma_levels = [0.05, 0.10, 0.15]
#         colors = ['#762a83', '#af8dc3', '#e7d4e8']
#         cluster_colors = ['#ef8a62','#b2182b','#2166ac','#67a9cf','#d1e5f0']

#         for sigma, color in zip(sigma_levels, colors):
#             ax1.fill_between(x, x - sigma, x + sigma, color=color, alpha=0.2,
#                              label=f'±{sigma:.2f} $\\mathrm{{M}}_{{\\odot}}$')

#         for run, color in zip(self.runs, cluster_colors):
#             df = run["df"]
#             label = run["label"]
#             cluster_type = run["cluster_type"]

#             if cluster_type == 'open':
#                 obs_mass = df["M"]
#                 obs_err = [df["M_lower_err"], df["M_upper_err"]]
#                 basta_err = [df["massfin_errm"], df["massfin_errp"]]
#             elif cluster_type == 'globular':
#                 obs_mass = df["howell_mass"]
#                 obs_err = [df["howell_mass_err"], df["howell_mass_err"]]
#                 basta_err = [df["massfin_errm"], df["massfin_errp"]]
#             else:
#                 continue

#             ax1.errorbar(
#                 obs_mass, df["massfin"],
#                 xerr=obs_err, yerr=basta_err,
#                 fmt='o', color=color, markeredgecolor='black', label=label, capsize=3
#             )

#             feh = df["FeH"]
#             feh_err = [df["FeH_errm"], df["FeH_errp"]]
#             age = df["age"] / 1e3
#             age_err = [df["age_errm"], df["age_errp"]]

#             ax2.scatter(
#                 age, feh, marker='D', color=color, edgecolor='black', label=label, zorder=2
#             )

#         feh_range = np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], 100)
#         ax2.fill_betweenx([-2.5,0.5], 7, 9, alpha=0.3, label='NGC6791 lit. range', color=cluster_colors[0], zorder=1)
#         ax2.fill_betweenx([-2.5,0.5], 2.3, 2.7 , alpha=0.3, label='NGC6819 lit. range', color=cluster_colors[1], zorder=1)
#         ax2.fill_betweenx([-2.5,0.5], 11.5, 13, alpha=0.3, label='M4 lit. range', color=cluster_colors[2], zorder=1)
#         ax2.fill_betweenx([-2.5,0.5], 12, 13, alpha=0.3, label='M80 lit. range', color=cluster_colors[3], zorder=1)
#         ax2.fill_betweenx([-2.5,0.5], 11, 13, alpha=0.3, label='M19 lit. range', color=cluster_colors[4], zorder=1)

#         ax2.set_ylim(-2.5,0.5)

#         ax1.set_xlabel(r"Literature seismic mass [$\mathrm{M}_{\odot}$]")
#         ax1.set_ylabel(r"BASTA seismic mass [$\mathrm{M}_{\odot}$]")
#         ax1.set_xlim(0.4, 2.2)
#         ax1.set_ylim(0.4, 2.2)
#         ax1.grid(True)

#         ax2.set_ylabel(r"[Fe/H]")
#         ax2.set_xlabel(r"Age [Gyr]")
#         ax2.grid(True)

#         # Move legends below each subplot
#         ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
#         ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

#         plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for legends
#         plt.savefig('results.png',format='png',dpi=400)


if __name__ == "__main__":

    configs = [("M4 (Howell+2022)", "../output/M4/results.ascii", "globular")]

    validator = PostProcessingValidator(
        configs, "literatureM_openClusters.dat", "literatureM_globularClusters.dat"
    )

    validator.plot_residuals()
