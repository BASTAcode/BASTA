# M4 Red Giant Example

Demonstration of a BASTA run using red giants in the globular cluster M4, with post-processing filters on final mass and age.

Ultimately, it is the user's responsibility to assess the physical plausibility of their results. This feature helps streamline that process by giving users greater flexibility to filter their best-fit solutions and inspect the consequences.

## Summary

We add functionality to apply user-defined bounds on final mass and age when selecting the best-fit model in BASTA. These bounds are not applied as priors, but are enforced after the full posterior is computed during result selection.

Note: BASTA includes isochrone grids that extend beyond the (current) universe age limit of 13.8 Gyr, reaching up to 16 Gyr. This design choice helps mitigate edge effects in the posterior distribution (see Section 3.1 of the BASTA II paper) by avoiding artificial truncation near hard boundaries when sampling old stellar populations. Although such extended models are useful during posterior computation, the resulting best-fit solutions may not be physically plausible in a cosmological context. By applying mass and age filters after the posterior has been computed, this feature preserves the statistical robustness of the sampling while allowing users to restrict final results to physically meaningful ranges.

bastamain.py has been modified to:

- Check if the best-fit model violates user-defined bounds on age and massfin.

- If it does, select the next-best model within bounds.

- If no valid model remains:

- (i) Return the original best fit (default), or

- (ii) Skip output entirely if "strict" is set to True.

Users can specify filters in the XML input file (BLOCK 2g). To apply no filtering:

    model_bounds = None

To apply filtering, define any combination of:

    "massfin": {"min": ..., "max": ...}

    "age": {"min": ..., "max": ...}

    "strict": controls behavior if no valid model is found

        False (or omitted): fallback to best model outside bounds

        True: skip model entirely if no valid solution found

        "mass", "age", or "both": enforce strict filtering on specific parameter(s)

Example:

    model_bounds = {
        "age": {"min": 0.01, "max": 13.8},
        "strict": "True",
    }

## Data

- Seismic input: Howell et al. (2022)

- Spectroscopic input: APOGEE DR17 (Abdurro’uf et al. 2022)

## Configuration

1. Stellar Grid

Download the BaSTI isochrone grid if not already available:

    BASTAdownload iso

2. Input Files

We provide three XML input files in run_feature/input_files/, built from templates in input_files/templates/:

    input_M4_nofilter.xml — no filtering (default behavior)

    input_M4_agefilter.xml — only age filtered (age ≤ 13.8 Gyr)

    input_M4_bothfilters.xml — age and mass filtered (mass ≤ 2.0 M$_\odot$)

3. Run All Three

Use the included shell script to run the three variants in series:

    chmod +x run_basta_M4.sh
    ./run_basta_M4.sh

For a typical laptop (4-core CPU), the full script takes approx. 11 mins.

## Output

Each BASTA run creates an output directory:

    run_feature/running_files/output/M4_nofilter
    run_feature/running_files/output/M4_agefilter
    run_feature/running_files/output/M4_bothfilters

Each contains:

    Posterior .json files (one per star)

    ASCII summary file: results.ascii

## Analysis & Plotting

The directory plot_results/ contains:

    literatureM_globularClusters.dat — literature seismic mass references

    plotting_notebook.ipynb — interactive example notebook

    plot_results.py — core Python class for post-processing and plotting
