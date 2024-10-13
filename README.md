# Matrix-Matrix Multiplication Optimizations

This project implements high-performance dense and sparse matrix-matrix multiplication in C, leveraging multi-threading, SIMD instructions, and cache optimization techniques to efficiently handle large-scale matrices. The codebase is organized into the `src/` directory for source files, `scripts/` for automation scripts, and `data/` for storing experimental results. A Makefile is provided to streamline the build process, ensuring easy compilation of the project components.

To install and run the program, follow these steps:

1. **Clone the Repository and Build the Project:**

2. **Run Experiments Using the Bash Script:**

    Navigate to the `scripts/` directory, make the script executable, and execute it to perform matrix multiplication experiments with various configurations:

    ```bash
    cd scripts
    chmod +x run_experiments.sh
    ./run_experiments.sh
    ```

    This script automates running the program with different matrix sizes, sparsity levels, and optimization settings, logging the results to `data/results.csv`.

3. **Manual Execution (Optional):**

    Alternatively, you can manually run the program with specific parameters:

    ```bash
    ./src/matrix_multiply [Matrix_Size] [Sparsity_A] [Sparsity_B] [MT] [Threads] [SIMD] [Cache_Opt]
    ```

    **Example:**

    ```bash
    ./src/matrix_multiply 1000 1.0 0.1 On 4 On Off
    ```

    This command runs a 1000x1000 matrix multiplication with 1% sparsity for Matrix A and 0.1% for Matrix B, enabling multi-threading with 4 threads and SIMD optimizations while disabling cache optimization. Results will be saved in `data/results.csv` for analysis.

---
