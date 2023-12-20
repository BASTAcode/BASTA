import argparse
import numpy as np
from basta.xml_run import run_xml


def main():
    """Main"""
    # Setup argument parser
    helptext = "BASTA -- Run the BAyesian STellar Algorithm"
    parser = argparse.ArgumentParser(description=helptext)

    # Add positional argument (name of inputfile)
    parser.add_argument("inputfile", help="The XML input file to run")

    # Add optional argument: Debugging output
    parser.add_argument(
        "--debug", action="store_true", help="Additional output for debugging."
    )

    # Add optional argument: Extra text output for debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional text output. A lot!" " Combine with the debug flag.",
    )

    # Add optional argument: Experimental features
    parser.add_argument(
        "--experimental", action="store_true", help="Enable experimental features."
    )

    # Add optional argument: Validation mode
    parser.add_argument(
        "--validation",
        action="store_true",
        help="DO NOT USE unless making validation runs.",
    )

    # Add optional argument: Set random seed
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Set random seed to ensure deterministic behavior for debugging",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed if given
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(5000)

    np.random.seed(seed)

    # Flag the user if running in validation mode
    if args.validation:
        print("\n*** Running in VALIDATION MODE ", end="")

    # Call BASTA
    run_xml(
        vars(args)["inputfile"],
        seed=seed,
        debug=args.debug,
        verbose=args.verbose,
        experimental=args.experimental,
        validationmode=args.validation,
    )
