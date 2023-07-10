First execute the following code to initialize the CUDA environment:
module load cuda/11.3.1

To execute the vector addition cuda file, run the following code:
nvcc vectorAddition.cu -o vectorAddition
./vectorAddition

To execute the matrix multiplication cuda file, run the following code:
nvcc matrixMult.cu -o matrixMult
./matrixMult

To execute the tiled matrix multiplication cuda file, run the following code:
nvcc tileMult.cu -o tileMult
./tileMult

Results are written to terminal. 
