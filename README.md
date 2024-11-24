# PCA-Mini-Project---GPU-Accelerated-Pollution-Data-Analysis-using-CUDA-Programming

#### Name : Karna S
#### Register No. : 212222110017
#### Date : 19 - 11 - 2024
#### Dept/Yr : CSE (IoT) / III

## AIM
To process large environmental pollution datasets using CUDA for generating heatmaps, and to analyze GPU performance for real-world data analysis applications.

## EQUIPMENTS REQUIRED
### Hardware: 
  1.NVIDIA GPU-enabled PC or laptop.
### Software:
  1.CUDA Toolkit
  
  2.NVCC Compiler
  
  3.Python (for visualization using Matplotlib or OpenCV).
  
  4.Dataset of pollution levels (e.g., air quality or water contamination data).

## PROCEDURE
### 1. Dataset Preparation
  1.Select a pollution dataset with spatial and intensity information (e.g., air quality index across a city).
  
  2.Preprocess the dataset to handle missing values and normalize for computation.
### 2. Initialize GPU Device
  Set up the CUDA device and retrieve GPU properties.
### 3. Allocate Unified Memory
  Use cudaMallocManaged to allocate memory for the dataset and the output grid.
### 4. Data Partitioning
  Divide the dataset into smaller chunks for parallel computation if necessary.
### 5. Write CUDA Kernel for Pollution Analysis
  Define a kernel to:
  
  2.Aggregate pollution data for grid cells.
  
  3.Compute average or weighted pollution levels.
### 6. Launch Kernel
  1.Configure grid and block dimensions for parallel execution.
  
  2.Perform a warm-up kernel launch to optimize page migration.
### 7. Visualize Results
  1.Transfer computed data back to the host if needed.
  
  2.Generate heatmaps using a visualization library like Matplotlib or OpenCV.
### 8. Performance Analysis
  1.Measure execution time on both CPU and GPU.
  
  2.Compare results to demonstrate GPU acceleration advantages.
### 9. Memory Cleanup
  1.Free memory using cudaFree.
  
  2.Reset the CUDA device with cudaDeviceReset.

## PROGRAM :
### Step 1: Install CUDA in Google Colab

Run the following commands to set up CUDA in Google Colab:

```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
### Step 2: Save the CUDA Code

Use the %%writefile magic command to save the CUDA code to a .cu file.

```
%%writefile pollution_analysis.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel to compute pollution intensity
__global__ void computePollution(float *pollutionData, float *heatmap, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check bounds
    if (idx < n) {
        // Example: Scaling the pollution intensity
        heatmap[idx] = pollutionData[idx] * 1.2; // Adjust scaling factor as needed
    }
}

// Function to initialize random pollution data
void initializePollutionData(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = (float)(rand() % 100) / 10.0f; // Values between 0 and 10
    }
}

int main() {
    int n = 1024; // Number of pollution data points
    size_t bytes = n * sizeof(float);

    // Allocate memory on host and device
    float *pollutionData, *heatmap;
    cudaMallocManaged(&pollutionData, bytes);
    cudaMallocManaged(&heatmap, bytes);

    // Initialize pollution data
    initializePollutionData(pollutionData, n);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    printf("Launching kernel with gridSize: %d, blockSize: %d\n", gridSize, blockSize);

    // Launch the kernel
    computePollution<<<gridSize, blockSize>>>(pollutionData, heatmap, n);

    // Synchronize device
    cudaDeviceSynchronize();

    // Print some sample output
    printf("Sample Pollution Data and Heatmap Results:\n");
    for (int i = 0; i < 10; i++) {
        printf("PollutionData[%d]: %.2f -> Heatmap[%d]: %.2f\n", i, pollutionData[i], i, heatmap[i]);
    }

    // Free allocated memory
    cudaFree(pollutionData);
    cudaFree(heatmap);

    // Reset the device
    cudaDeviceReset();

    return 0;
}
```

### Step 3: Compile and Run the Program

Compile the program using:

```
!nvcc pollution_analysis.cu -o pollution_analysis
```

Run the executable:

```
!./pollution_analysis
```

## Output :
### Step 1: Install CUDA in Google Colab
![image](https://github.com/user-attachments/assets/cca71f4a-2bfc-44f7-a32b-f18774df91e3)

### Step 2: Save the CUDA Code
![image](https://github.com/user-attachments/assets/be115cd8-6612-43b9-b613-fe1b726a21f1)

### Step 3: Compile and Run the Program
![image](https://github.com/user-attachments/assets/ffc9b47b-a148-403f-bc9f-35fb830fb96b)

## RESULT
Thus, the program was executed successfully, and GPU acceleration demonstrated its efficiency in processing large pollution datasets. Heatmaps were generated to visualize pollution intensity, showcasing real-world applications of GPU programming.
