#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

static void ping_pong(void *buffer, int count, MPI_Datatype dtype, MPI_Comm comm)
{
  /* Implement a ping pong.
   *
   * rank 0 should send count bytes from buffer to rank 1
   * rank 1 should then send the received data back to rank 0
   *
   */
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    MPI_Send(buffer, count, dtype, 1, 0, comm);
    MPI_Recv(buffer, count, dtype, 1, 0, comm, MPI_STATUS_IGNORE);


  } else if (rank == 1){
    MPI_Recv(buffer, count, dtype, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Send(buffer, count, dtype, 0, 0, comm);

  }




}

int main(int argc, char **argv)
{
  int nbytes, rank;
  char *buffer;
  double start, end;
  MPI_Comm comm;

  MPI_Init(&argc, &argv);

  comm = MPI_COMM_WORLD;
  nbytes = argc > 1 ? atoi(argv[1]) : 1;

  buffer = calloc(nbytes, sizeof(*buffer));
  
  /* Figure out how long one ping-pong takes */
  start = MPI_Wtime();
  ping_pong(buffer, nbytes, MPI_CHAR, comm);
  end = MPI_Wtime();

  /* Run for approximately 1 second */
  int niterations = (int)(1.0/(end - start));
  if (niterations < 1) {
    niterations = 1;
  }


  start = MPI_Wtime();
  for (int i = 0; i < niterations; i++) {
    ping_pong(buffer, nbytes, MPI_CHAR, comm);
  }
  end = MPI_Wtime();
  MPI_Comm_rank(comm, &rank);
  if (rank==0) { 
    printf("Ping-pong took %f seconds\n", (end-start)/niterations);
  }

  free(buffer);

  MPI_Finalize();
  return 0;
}
