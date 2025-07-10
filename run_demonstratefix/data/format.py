input_file = "M4.ascii"
output_file = "M4_.ascii"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        # Replace any sequence of whitespace with a single tab
        line_tabbed = "\t".join(line.strip().split())
        f_out.write(line_tabbed + "\n")

print(f"Saved tab-separated file to {output_file}")
