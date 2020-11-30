#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void print_matrix(const double *A, int n, MPI_Comm comm)
{
  /* Implement printing of A by gathering to rank 0 and printing there. */
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  const double *sendbuf = A;
  double *recvbuf=NULL;
  if (rank == 0) {
    recvbuf = malloc(size*n*sizeof(double));
  }
  MPI_Gather(sendbuf, n, MPI_DOUBLE, recvbuf, n, MPI_DOUBLE, 0, comm);
  if (rank == 0) {
    for (int row = 0; row < size; row++){
      for (int val = 0; val < n; val++){
        int index = row * n + val;
        printf("%.2f\t", recvbuf[index]);
      };
      printf("\n");
    }
  }
}

static void transpose_matrix(const double *A, double *AT, int n, MPI_Comm comm)
{
  /* Implement AT <- transpose(A) (MPI_Alltoall may be helpful) */ 
  MPI_Alltoall(A, 1, MPI_DOUBLE, AT, 1, MPI_DOUBLE, comm);

  // Note to self: the send/recv count in this instance is the count to/from
  // EACH process, not overall...

}

static void matrix_vector_product(const double *A, const double *x, double *y, int n, MPI_Comm comm)
{
  /* Implement y <- Ax, it might be useful to tranpose A first (think
     about the data distribution). */
  double *AT = NULL;
  double *ATT=NULL;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  AT = malloc(4*size * sizeof(*AT));
  ATT = malloc(4*size * sizeof(*ATT));

  transpose_matrix(A, AT, size, comm);
  for (int i=0; i<n; i++){
    AT[i] *= x[0];
  }
  transpose_matrix(AT, ATT, size, comm);
  for (int i=0; i<n; i++){
    y[0] += ATT[i];
  }
  free(AT);
  free(ATT);
}

static void print_vector(const double *x, MPI_Comm comm)
{
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  if (rank == 0) {
    double tmp;
    for (int i = 0; i < size; i++) {
      if (i != rank) {
        MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE);
      } else {
        tmp = x[0];
      }
      printf("%g\n", tmp);
    }
    printf("\n");
  } else {
    MPI_Ssend(x, 1, MPI_DOUBLE, 0, 0, comm);
  }
}

int main(int argc, char **argv)
{
  int rank;
  int size;
  double *A = NULL;
  double *AT = NULL;
  double x;
  double y;
  MPI_Comm comm;
  MPI_Init(&argc, &argv);

  comm = MPI_COMM_WORLD;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  A = malloc(4*size * sizeof(*A));
  AT = malloc(4*size * sizeof(*AT));

  x = rank + 1;
  for (int i = 0; i < 4*size; i++) {
    A[i] = 4*size*rank + i;
  }

  if (rank == 0) {
    printf("Matrix A is:\n\n");
  }
  print_matrix(A, size, comm);

  transpose_matrix(A, AT, size, comm);

  if (rank == 0) {
    printf("\nMatrix A^T is:\n\n");
  }

  print_matrix(AT, size, comm);

  if (rank == 0) {
    printf("\nVector x is:\n\n");
  }

  print_vector(&x, comm);
  matrix_vector_product(A, &x, &y, size, comm);

  if (rank == 0) {
    printf("\nVector y <- Ax is:\n\n");
  }

  print_vector(&y, comm);
  free(A);
  free(AT);
  MPI_Finalize();
}
