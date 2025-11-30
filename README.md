# High Dimensional Search
The following command-line arguments are supported:
- `-t / --traing_data`: Path to the zip file containing training data. This argument will also automatically trigger training. The trained model will be stored in a file with the same name and extension `.model`.
- `-m / --model`: Path to a trained model file. This argument is required if `-t` is not set in the same run and an indexing algorithm is used.
- `-q / --quantize`: Use quantization for indexing.
- `--lsh`: Use Locality Sensitive Hashing (LSH) for indexing.
- `--annoy`: Use the Annoy library for indexing.