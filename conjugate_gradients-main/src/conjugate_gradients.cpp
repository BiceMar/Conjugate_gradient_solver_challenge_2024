#include <cstdio>
#include <cstdlib>
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

    // Calculate the number of rows this process should handle
    if(total_rows % mpi_size != 0){
        (rank != mpi_size - 1) ? num_rows_local = (total_rows / mpi_size) : num_rows_local = (total_rows / mpi_size) + (total_rows % mpi_size);
        fseek(file, (rank * (total_rows / mpi_size) * num_cols_local) * sizeof(double), SEEK_CUR);
    }
    else{
        num_rows_local = total_rows / mpi_size;
        fseek(file, (rank * num_rows_local * num_cols_local) * sizeof(double), SEEK_CUR);
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
    double result = 0.0;
    #pragma omp parallel for shared(x, y) schedule(static) reduction(+:result) 
    for(size_t i = 0; i < size; i++) {
        result += x[i] * y[i];
    }

    return result;
}

void axpbyP(double alpha, const double * x, double beta, double * y, size_t size)
{
    #pragma omp parallel for shared(x, y) schedule(static) 
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void gemvP(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    #pragma omp parallel for schedule(static)
    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        #pragma omp simd reduction(+:y_val)
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}


void conjugate_gradients(const double * A, const double * b, double * x, size_t local_size, size_t total_rows, size_t max_iters, double rel_error)
{   
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); 

    MPI_File file;
    MPI_Status status;

    size_t num_iters;
    double alpha, beta, rr, rr_new; double bb; 
    double *tmp1 = new double; 
    double *tmp2 = new double; 
    double * p_temp = new double[local_size]; 
    double * p = new double[total_rows];
    double * Ap_temp = new double[local_size]; 
    double * Ap = new double[total_rows];
    double * r = new double[local_size];

    #pragma omp parallel for
    for(size_t i = 0; i < local_size; i++)
    {
        x[i] = 0.0; 
        r[i] = b[i]; 
        p_temp[i] = b[i];
    }

    int * rows_per_processes = new int[mpi_size];
    int * row_offsets = new int[mpi_size];

    if(total_rows % mpi_size != 0)
        #pragma omp parallel for
        for(int i = 0; i < mpi_size; i++){
            if(i != mpi_size - 1){
                rows_per_processes[i] = (total_rows / mpi_size);
            }else{
                rows_per_processes[i] = (total_rows / mpi_size) + (total_rows % mpi_size);
            }
        }else{
            #pragma omp parallel for
            for(int i = 0; i < mpi_size; i++)
                rows_per_processes[i] = total_rows / mpi_size;
        }

    int offset = 0; 
    row_offsets[0] = 0;
    for (int i = 1; i < mpi_size; i++) 
    {
        row_offsets[i] = offset + rows_per_processes[i-1];
        offset += rows_per_processes[i];
    }

    bb = dotP(b, b, local_size);
    MPI_Allreduce(&bb, &bb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    rr = bb;

    MPI_Allgatherv(p_temp, local_size, MPI_DOUBLE, p, rows_per_processes, row_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {   
        gemvP(1.0, A, p, 0.0, Ap_temp, local_size, total_rows);
        MPI_Allgatherv(Ap_temp, local_size, MPI_DOUBLE, Ap, rows_per_processes, row_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
        
        *tmp1 = dotP(p_temp, Ap_temp, local_size);
        MPI_Allreduce(tmp1, tmp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        alpha = rr / *tmp2;
        axpbyP(alpha, p_temp, 1.0, x, local_size);
        axpbyP(-alpha, Ap_temp, 1.0, r, local_size);

        *tmp1 = dotP(r, r, local_size);
        MPI_Allreduce(tmp1, tmp2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        rr_new = *tmp2;
        beta = rr_new / rr;
        rr = rr_new;

        if(std::sqrt(rr / bb) < rel_error)  
            break; 

        axpbyP(1.0, r, beta, p_temp, local_size);
        MPI_Allgatherv(p_temp, local_size, MPI_DOUBLE, p, rows_per_processes, row_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    if(rank == 0)
    {
        if(num_iters <= max_iters)
            printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        else
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    
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
    delete[] Ap_temp; 
    delete[] p_temp; 
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
    size_t rhs_rows;
    size_t rhs_cols;
    size_t size; 

    // Read matrix

    if(rank == 0)
        printf("Reading matrix from file\n\n");

    bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows_local, &matrix_cols);
    
    if(!success_read_matrix){
        fprintf(stderr, "Failed to read matrix\n");
        return 1;
    }

    // Read rhs

    if(rank == 0)
        printf("Reading right hand side from file\n\n");

    bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
    
    if(!success_read_rhs){
        fprintf(stderr, "Failed to read rhs\n");
        return 2;
    }

    if(rank == 0)
        printf("Done\n\n");

    // Controls  

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
    
    size = matrix_rows_local; 

    // Solve the sistem

    double * sol = new double[size];
    double start_time = MPI_Wtime();

    conjugate_gradients(matrix, rhs, sol, size, matrix_cols, max_iters, rel_error);

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if(rank == 0)
        printf("Finished successfully. Time taken to solve the sistem of size %d: %f seconds", size, elapsed_time);
    
    delete[] matrix; 
    delete[] rhs; 
    delete[] sol;

    MPI_Finalize();

    return 0;
}
