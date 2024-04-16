#!/bin/bash

# Create the directory if it doesn't already exist
mkdir -p ranked

# Change into the directory
cd ranked

# Loop from 0 to 47
for i in {0..47}
do
    # Construct the URL
    url="https://pub-8978012207224952a747e641910bcb1c.r2.dev/ranked/Tabula_Sapiens_ranked_${i}.h5ad"

    # Download the file using wget
    wget $url
done

# Change back to the original directory
cd ..
