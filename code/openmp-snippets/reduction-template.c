#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>

int main(int argc, char **argv)
{
  const size_t N = argc > 1 ? strtoul(argv[1], NULL, 10) : 1024u;
  double *a = malloc(N * sizeof(*a));
  double *b = malloc(N * sizeof(*b));

  printf("Computing dot product with %lu entries\n", N);
  /* Intialise with some values */
#pragma omp parallel for default(none) schedule(static) shared(a, b, N)  
  for (size_t i = 0; i < N; i++) {
    a[i] = i+1;
    b[i] = (-1)*(i%2) * i;
  }

  /* Implement a parallel dot product using
     *
     * 1. The approach of reduction-hand.c
     * 2. the reduction clause
     * 3. critical sections to protect the shared updates
     * 4. atomics to protect the shared updates.
     * 
     */
  
  double start;
  double stop;

  /**** Implement by hand ****/

  start = omp_get_wtime();
  double dotabparallel = 0;
  double *dotlocal = NULL;
#pragma omp parallel default(none) shared(a, b, N, dotabparallel, dotlocal)
  {
    int tid = omp_get_thread_num();
  #pragma omp single
    dotlocal = calloc(omp_get_num_threads(), sizeof(*dotlocal));
  
  #pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      dotlocal[tid] += a[i]*b[i];
    }

  #pragma omp single nowait
    for (int i = 0; i < omp_get_num_threads(); i++) {
      dotabparallel += dotlocal[i];
    }
  }
stop = omp_get_wtime();
printf("Parallel by hand \t\ta.b = %.9g; took %.4g seconds\n", dotabparallel, (stop - start));
free(dotlocal);


/**** Implement reduction clause version ****/

start = omp_get_wtime();
double dotreductionclause = 0;
#pragma omp parallel default(none) shared(a, b, N) reduction(+:dotreductionclause)
{
  double dotlocal = 0;
#pragma omp for schedule(static)
  for (int i=0; i<N; i++) {
    dotlocal += a[i]*b[i];
  }

  dotreductionclause += dotlocal;
}
stop = omp_get_wtime();
printf("Parallel reduction clause \ta.b = %.9g; took %.4g seconds\n", dotreductionclause, (stop-start));


/**** Implement critical section version ****/

start = omp_get_wtime();
double dotcritical = 0;
#pragma omp parallel default(none) shared(a, b, N, dotcritical)
{
  double dotlocal = 0;
#pragma omp for schedule(static)
  for (int i=0; i<N; i++) {
    dotlocal += a[i]*b[i];
  }
#pragma omp critical
  dotcritical += dotlocal;
}
stop = omp_get_wtime();
printf("Parallel critical \t\ta.b = %.9g; took %.4g seconds\n", dotcritical, (stop-start));


/**** Implement atomics version ****/

start = omp_get_wtime();
double dotatomics = 0;
#pragma omp parallel default(none) shared(a, b, N, dotatomics)
{
  double dotlocal = 0;
#pragma omp for schedule(static)
  for (int i=0; i<N; i++) {
    dotlocal += a[i]*b[i];
  }
#pragma omp atomic
  dotatomics += dotlocal;
}
stop = omp_get_wtime();
printf("Parallel atomics \t\ta.b = %.9g; took %.4g seconds \n", dotatomics, (stop-start));




  free(a);
  free(b);
  return 0;
}
