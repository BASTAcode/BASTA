#!/bin/bash

# Exit if any command fails
set -e

echo "Running BASTA with no filters..."
BASTArun running_files/input_M4_nofilter.xml

echo "Running BASTA with age filter..."
BASTArun running_files/input_M4_agefilter.xml

echo "Running BASTA with age + mass filters..."
BASTArun running_files/input_M4_bothfilters.xml

echo "All BASTA runs completed successfully."
