#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <mpi.h>

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{   
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 
    double * matrix;
    size_t total_rows, num_rows_local, num_cols_local;
    FILE * file = fopen(filename, "rb");

    // Read the total number of rows and columns from the file
    fread(&total_rows, sizeof(size_t), 1, file);
    fread(&num_cols_local, sizeof(size_t), 1, file);
    fseek(file, (rank * (total_rows / mpi_size) * num_cols_local) * sizeof(double), SEEK_CUR);

    // Calculate the number of rows this process handles
    if(total_rows % mpi_size != 0){
        (rank != mpi_size - 1) ? num_rows_local = (total_rows / mpi_size) : num_rows_local = (total_rows / mpi_size) + (total_rows % mpi_size);
    }
    else{
        num_rows_local = total_rows / mpi_size;
    }
    
    matrix = new double[num_rows_local * num_cols_local];
    
    // Read the appropriate number of doubles from the file based on the process rank
    fread(matrix, sizeof(double), num_rows_local * num_cols_local, file);

    *num_cols_out = num_cols_local;
    *num_rows_out = num_rows_local;
    *matrix_out = matrix;

    fclose(file);

    return true;
}

void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

double dotP(const double * x, const double * y, size_t size) {
    // Initialize the result and the variable to hold the sub-products
    double result = 0.0;
    double sub_prod = 0.0;

    // Parallelize the computation of the dot product
    #pragma omp parallel for shared(x, y) reduction(+:sub_prod) 
    for(size_t i = 0; i < size; i++) {
        // Accumulate the product of corresponding elements
        sub_prod += x[i] * y[i];
    }
    
    // Use MPI to reduce (sum up) all the partial dot products into 'result'
    MPI_Allreduce(&sub_prod, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return result;
}


void axpbyP(double alpha, const double * x, double beta, double * y, size_t size)
{
    #pragma omp parallel for shared(x, y) 
    for(size_t i = 0; i < size; i++)
    {
        // Perform the operation y = alpha * x + beta * y for each element
        y[i] = alpha * x[i] + beta * y[i];
    }
}


void gemvP(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    // Parallelize over the rows of the matrix
    #pragma omp parallel for private(y_val)
    for(size_t r = 0; r < num_rows; r++)
    {
        // Initialize the accumulator for this row
        double y_val = 0.0;

        //#pragma omp simd reduction(+:y_val)
        for(size_t c = 0; c < num_cols; c++)
        {
            // Compute the dot product of the row of A and vector x, scaled by alpha
            y_val += alpha * A[r * num_cols + c] * x[c];
        }

        // Update y by adding the scaled result to the scaled original y values
        y[r] = beta * y[r] + y_val;
    }
}


// `A` is the matrix, `b` is the right-hand side vector, `x` is the solution vector.
// `local_size` is the number of rows of `A` handled by this process, `total_rows` is the total number of rows in `A`.
void conjugate_gradients(const double * A, const double * b, double * x, size_t local_size, size_t total_rows, size_t max_iters, double rel_error)
{
    int rank, mpi_size; // MPI process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_File file;
    MPI_Status status;

    size_t num_iters; // Counter for the number of iterations
    double alpha, beta, rr, rr_new, bb; // Scalars for algorithm steps
    double *tmp1 = new double; // Temporary storage for dot product results
    double *tmp2 = new double; // Temporary storage for reduced dot product results
    double * p = new double[total_rows]; // Global search direction vector
    double * p_local = new double[local_size]; // Local search direction vector
    double * Ap_local = new double[local_size]; // Local matrix-vector product result
    double * Ap = new double[total_rows]; // Global matrix-vector product result
    double * r = new double[local_size]; // Local residual vector

    // Initialize x to zero and r and p_tmp to b locally for each process
    #pragma omp parallel for
    for(size_t i = 0; i < local_size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p_local[i] = b[i];
    }

    // Compute the distribution of rows across processes
    int * rows_per_processes = new int[mpi_size]; // Number of rows handled by each process
    int * row_offsets = new int[mpi_size]; // Starting offset of rows for each process

    // Adjust the distribution if total_rows is not divisible evenly by mpi_size
    if(total_rows % mpi_size != 0){
        #pragma omp parallel for
        for(int i = 0; i < mpi_size; i++)
            (i != mpi_size - 1) ? rows_per_processes[i] = (total_rows / mpi_size) : rows_per_processes[i] = (total_rows / mpi_size) + (total_rows % mpi_size);
    }
    else{
        #pragma omp parallel for
        for(int i = 0; i < mpi_size; i++)
            rows_per_processes[i] = total_rows / mpi_size;
    }

    // Calculate row offsets based on rows_per_processes
    int offset = 0;
    row_offsets[0] = 0;
    #pragma omp parallel for
    for (int i = 1; i < mpi_size; i++)
    {
        row_offsets[i] = offset + rows_per_processes[i-1];
        offset += rows_per_processes[i];
    }

    // Compute b*b and reduce it across all processes
    bb = dotP(b, b, local_size);
    rr = bb; 

    // Gather initial search directions from all processes
    MPI_Allgatherv(p_local, local_size, MPI_DOUBLE, p, rows_per_processes, row_offsets, MPI_DOUBLE, MPI_COMM_WORLD);

    // Main iteration loop
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemvP(1.0, A, p, 0.0, Ap_local, local_size, total_rows);

        // Compute the dot product of p and Ap and reduce the result
        *tmp2 = dotP(p_local, Ap_local, local_size);

        // Update alpha, x, and r using the results
        alpha = rr / *tmp2;

        axpbyP(alpha, p_local, 1.0, x, local_size);
        axpbyP(-alpha, Ap_local, 1.0, r, local_size);

        // Compute the new residual norm and reduce the result
        *tmp2 = dotP(r, r, local_size);

        rr_new = *tmp2; // Update the residual norm
        beta = rr_new / rr; // Update beta
        rr = rr_new; // Prepare for next iteration

        // Check for convergence
        if(std::sqrt(rr / bb) < rel_error)
            break; // Exit loop if converged

        // Update the search direction and gather the result from all processes
        axpbyP(1.0, r, beta, p_local, local_size);
        MPI_Allgatherv(p_local, local_size, MPI_DOUBLE, p, rows_per_processes, row_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
        if(num_iters <= max_iters)
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        else
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    
    // parallel write on file
    if(num_iters <= max_iters)
    {
        MPI_File_open(MPI_COMM_WORLD, "io/sol_mpi.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        if(total_rows % mpi_size != 0)
            MPI_File_seek(file, rank * (total_rows / mpi_size) * sizeof(double), MPI_SEEK_SET);
        else
            MPI_File_seek(file, rank * local_size * sizeof(double), MPI_SEEK_SET); 
        MPI_File_write(file, x, local_size, MPI_DOUBLE, &status);
        MPI_File_close(&file);
    }

    delete[] r; 
    delete[] p; 
    delete[] Ap; 
    delete[] tmp1; 
    delete[] tmp2;
    delete[] Ap_local; 
    delete[] p_local; 
}

int main(int argc, char ** argv)
{   
    // MPI 
    int rank, mpi_size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 

    // Variables for the conjugate gradient method
    size_t max_iters = 1000;
    double rel_error = 1e-9;

    const char * input_file_matrix = "io/matrix.bin"; 
    const char * input_file_rhs = "io/rhs.bin"; 
    const char * output_file_sol = "io/sol_mpi.bin";

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    if(rank == 0){
        printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
        printf("All parameters are optional and have default values\n");
        printf("\n");

        printf("Command line arguments:\n");
        printf("  input_file_matrix: %s\n", input_file_matrix);
        printf("  input_file_rhs:    %s\n", input_file_rhs);
        printf("  output_file_sol:   %s\n", output_file_sol);
        printf("  max_iters:         %d\n", max_iters);
        printf("  rel_error:         %e\n", rel_error);
        printf("\n");
    }

    double * matrix;
    size_t matrix_rows_local;
    size_t matrix_cols;
    size_t total_rows_rhs;
    double * rhs;
    size_t rhs_rows, rhs_cols;
    size_t size, local_size; 

    // Read matrix
    if(rank == 0)
        printf("Reading matrix right hand side from file\n\n");

    bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows_local, &matrix_cols); 
    bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);   

    if(rank == 0)
        printf("Done\n\n");

    // Controls
    if(!success_read_matrix){
        fprintf(stderr, "Failed to read matrix\n");
        return 1;
    } 
    if(!success_read_rhs){
        fprintf(stderr, "Failed to read rhs\n");
        return 2;
    } 
    if(rhs_rows != matrix_rows_local)
    {
        fprintf(stderr, "Size of right hand side does not match the matrix\n");
        return 4;
    }
    if(rhs_cols != 1)
    {
        fprintf(stderr, "Right hand side has to have just a single column\n");
        return 5;
    }
    
    // Solve the sistem
    double * sol = new double[matrix_cols];
    double start_time = MPI_Wtime();

    conjugate_gradients(matrix, rhs, sol, matrix_rows_local, matrix_cols, max_iters, rel_error);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if(rank == 0)
        printf("Finished successfully. Time taken to solve the sistem of size %d: %f seconds", matrix_cols, elapsed_time);
    
    delete[] matrix; 
    delete[] rhs; 
    delete[] sol;

    MPI_Finalize();

    return 0;
}
