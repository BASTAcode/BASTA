import csv
import numpy as np

from basta import constants as bc


def make_tables():
    """
    Makes csv tables of constants.parameters.params, for the
    documentation, to be more readily readable by users.
    Divides into two tables: Photometric filters/magnitudes,
    and the rest (classic).
    """

    # Get all parameters, sort by filter and sort classic
    allparams = np.array(bc.parameters.params)
    filters = [f[0] for f in bc.extinction.R]
    clasparams = []
    filtparams = []
    for param in allparams:
        if param[0] in filters:
            filtparams.append(param)
        else:
            clasparams.append(param)
    clasparams = np.array(clasparams)[np.argsort([p[0].lower() for p in clasparams])]

    # Open table, prepare writer, write header
    parmfile = open("param_classic.csv", "w", newline="")
    parmwriter = csv.writer(parmfile, delimiter="|")
    parmwriter.writerow(["Key", "Unit", "Parameter", "Description"])

    # Write table
    for param in clasparams:
        # Reformat and write entry
        refparam = reformat_entry(param)
        parmwriter.writerow(refparam)

    # Close table
    parmfile.close()

    # Open table, prepare writer, write header
    filtfile = open("param_filters.csv", "w", newline="")
    filtwriter = csv.writer(filtfile, delimiter="|")
    filtwriter.writerow(["Key", "Unit", "Parameter", "Description"])

    # Write table
    for param in filtparams:
        # Reformat and write entry
        refparam = reformat_entry(param)
        filtwriter.writerow(refparam)

    # Close table file
    filtfile.close()


def reformat_entry(listpar):
    # Ignore color column
    lpar = list(listpar)[:-1]

    # None should just be empty entries
    if None in lpar:
        lpar[np.where(np.array(lpar) == None)[0][0]] = ""

    # Replace LaTeX math with reStructuredText math
    # Wrap remaining in \text
    if "$" in lpar[2]:
        string = ":math:`"
        splstr = lpar[2].split("$")
        for j, ss in enumerate(splstr):
            if j % 2 == 0 and ss != "":
                string += "\\text{" + ss + "}"
            elif ss != "":
                string += ss
        string += "`"

        lpar[2] = string

    # Sometimes it is also in description. Above changes
    # font family, the following only works if there are
    # ONLY ONE math section
    if "$" in lpar[3]:
        lpar[3] = lpar[3].replace("$", " :math:`", 1)
        lpar[3] = lpar[3].replace("$", "`", 1)

    # Wrap key in backquotes for highlighting
    lpar[0] = "``" + lpar[0] + "``"
    return lpar


if __name__ == "__main__":
    make_tables()
