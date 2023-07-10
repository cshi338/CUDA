import numpy as np
import matplotlib.pyplot as plt

vectorSize = []
cpu = []
gpu = []

with open("vectorAdd.txt") as file:
    lines = file.readlines()
    for line in lines:
        if "Vector Size of " in line:
            temp = line.split("Vector Size of ",1)[1]
            temp = temp.split('\n')[0]
            vectorSize.append(int(temp))
        elif "CPU" in line:
            temp = line.split("took ",1)[1]
            temp = temp.split(" seconds")[0]
            cpu.append(float(temp))
        elif "GPU" in line:
            temp = line.split("took ",1)[1]
            temp = temp.split(" seconds")[0]
            gpu.append(float(temp))


print("Size of Vector: " + str(vectorSize))
print("CPU Runtimes (sec): " + str(cpu))
print("GPU Runtimes (sec): " + str(gpu) + "\n")
plt.scatter(vectorSize, cpu, label = "CPU")
plt.plot(vectorSize, cpu)
plt.scatter(vectorSize, gpu, label = "GPU")
plt.plot(vectorSize, gpu)
plt.ylabel('Time to Complete Operation (sec)')
plt.xlabel('Vector Size')
plt.title('Vector Addition Execution Time vs Vector Size')
plt.legend()
plt.show()

vectorSize = []
cpu = []
gpu = []

with open("matrixMult.txt") as file:
    lines = file.readlines()
    for line in lines:
        if "NxN Matrix " in line:
            temp = line.split("N = ",1)[1]
            temp = temp.split('\n')[0]
            vectorSize.append(int(temp))
        elif "CPU" in line:
            temp = line.split("took ",1)[1]
            temp = temp.split(" seconds")[0]
            cpu.append(float(temp))
        elif "GPU" in line:
            temp = line.split("took ",1)[1]
            temp = temp.split(" seconds")[0]
            gpu.append(float(temp))


print("Size of N for NxN Matrix: " + str(vectorSize))
print("CPU Runtimes (sec): " + str(cpu))
print("GPU Runtimes (sec): " + str(gpu) + "\n")
plt.scatter(vectorSize, cpu, label = "CPU")
plt.plot(vectorSize, cpu)
plt.scatter(vectorSize, gpu, label = "GPU")
plt.plot(vectorSize, gpu)
plt.ylabel('Time to Complete Operation (sec)')
plt.xlabel('Size of N for NxN Matrix')
plt.title('Matrix Multiplication Execution Time vs Input Size')
plt.legend()
plt.show()


tileMults = [2, 4, 8, 16, 32]
for tileMult in tileMults:
    vectorSize = []
    cpu = []
    gpu = []
    with open("tileMult" + str(tileMult)+ ".txt") as file:
        lines = file.readlines()
        for line in lines:
            if "NxN Matrix" in line:
                temp = line.split("N = ",1)[1]
                #temp = temp.split('\n')[0]
                vectorSize.append(int(temp))
            elif "CPU" in line:
                temp = line.split("took ",1)[1]
                temp = temp.split(" seconds")[0]
                cpu.append(float(temp))
            elif "GPU" in line:
                temp = line.split("took ",1)[1]
                temp = temp.split(" seconds")[0]
                gpu.append(float(temp))

    print("Size of N for NxN Matrix: " + str(vectorSize) + ' with Tile Size = ' + str(tileMult))
    print("CPU Runtimes (sec): " + str(cpu))
    print("GPU Runtimes (sec): " + str(gpu) + "\n")
    plt.scatter(vectorSize, cpu, label = "CPU")
    plt.plot(vectorSize, cpu)
    plt.scatter(vectorSize, gpu, label = "GPU")
    plt.plot(vectorSize, gpu)
    plt.ylabel('Time to Complete Operation (sec)')
    plt.xlabel('Size of N for NxN Matrix')
    plt.title('Matrix Multiplication with Tile Size = ' + str(tileMult) + ' Execution Time vs Input Size')
    plt.legend()
    plt.show()
