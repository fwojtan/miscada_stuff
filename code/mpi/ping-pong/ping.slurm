#!/bin/bash -l
#
# Batch script for bash users
#
#SBATCH -n 2 #the number of cores we are using
#SBATCH -J pingpong #a name for our job
#SBATCH -o pingpong.%J.dump #the logfile
#SBATCH -e pingpong.%J.err #the error file
#SBATCH -p par7.q #the queue we want to use
#SBATCH -t 00:05:00 ## the maximum run time

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# Need to ensure the right MPI module is loaded -
# i.e. the same module which the program was compiled with.
# specify the modules you compiled the code with below
module purge #unload all modules
module load slurm/current
module load intelmpi/intel/2018.2
module load intel/xe_2018.2
module load gcc/9.3.0
module list #write a list of used modules to the outputfile
# Run the program
#add the correct command to run the program below
for n in 1000 100000 10000000 1000000000 100000000000 10000000000000
do
echo $n >> results.txt
mpirun ./pingpong $n >> results.txt
done
#print some info at the end
echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit