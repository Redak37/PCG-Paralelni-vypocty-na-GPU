/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Paralelní programování na GPU (PCG 2021)
 * Projekt c. 1 (cuda)
 * Login: xducho07
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Time measurement
  struct timeval t1, t2;

  if (argc != 10)
  {
    printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
    exit(1);
  }

  // Number of particles
  const int N           = std::stoi(argv[1]);
  // Length of time step
  const float dt        = std::stof(argv[2]);
  // Number of steps
  const int steps       = std::stoi(argv[3]);
  // Number of thread blocks
  const int thr_blc     = std::stoi(argv[4]);
  // Write frequency
  int writeFreq         = std::stoi(argv[5]);
  // number of reduction threads
  const int red_thr     = std::stoi(argv[6]);
  // Number of reduction threads/blocks
  const int red_thr_blc = std::stoi(argv[7]);

  // Size of the simulation CUDA grid - number of blocks
  const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
  // Size of the reduction CUDA grid - number of blocks
  const size_t reductionGrid  = (red_thr + red_thr_blc - 1) / red_thr_blc;

  // Log benchmark setup
  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);
  printf("threads/block: %d\n", thr_blc);
  printf("blocks/grid: %lu\n", simulationGrid);
  printf("reduction threads/block: %d\n", red_thr_blc);
  printf("reduction blocks/grid: %lu\n", reductionGrid);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
  writeFreq = (writeFreq > 0) ?  writeFreq : 0;


  t_particles particles_cpu;
  float4 *comOnGPUA;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                            FILL IN: CPU side memory allocation (step 0)                                          //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cudaMallocHost(&particles_cpu.pos_x, N * sizeof(float));
  cudaMallocHost(&particles_cpu.pos_y, N * sizeof(float));
  cudaMallocHost(&particles_cpu.pos_z, N * sizeof(float));
  cudaMallocHost(&particles_cpu.vel_x, N * sizeof(float));
  cudaMallocHost(&particles_cpu.vel_y, N * sizeof(float));
  cudaMallocHost(&particles_cpu.vel_z, N * sizeof(float));
  cudaMallocHost(&particles_cpu.weight, N * sizeof(float));
  cudaMallocHost(&comOnGPUA, sizeof(*comOnGPUA));

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                              FILL IN: memory layout descriptor (step 0)                                          //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                      Stride of two               Offset of the first
   *  Data pointer        consecutive elements        element in floats,
   *                      in floats, not bytes        not bytes
  */
  MemDesc md(
        particles_cpu.pos_x,        1,                      0,              // Postition in X
        particles_cpu.pos_y,        1,                      0,              // Postition in Y
        particles_cpu.pos_z,        1,                      0,              // Postition in Z
        particles_cpu.vel_x,        1,                      0,              // Velocity in X
        particles_cpu.vel_y,        1,                      0,              // Velocity in Y
        particles_cpu.vel_z,        1,                      0,              // Velocity in Z
        particles_cpu.weight,       1,                      0,              // Weight
        N,                                                                  // Number of particles
        recordsNum);                                                        // Number of records in output file

  // Initialisation of helper class and loading of input data
  H5Helper h5Helper(argv[8], argv[9], md);

  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  }
  catch (const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return -1;
  }


  t_particles p_in, p_out;
  float *comX, *comY, *comZ, *comW;
  int *lock;
  cudaStream_t stream0, stream1, stream2;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                  FILL IN: GPU side memory allocation (step 0)                                    //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cudaMalloc<float>(&p_in.pos_x, N * sizeof(float));
  cudaMalloc<float>(&p_in.pos_y, N * sizeof(float));
  cudaMalloc<float>(&p_in.pos_z, N * sizeof(float));
  cudaMalloc<float>(&p_in.vel_x, N * sizeof(float));
  cudaMalloc<float>(&p_in.vel_y, N * sizeof(float));
  cudaMalloc<float>(&p_in.vel_z, N * sizeof(float));
  cudaMalloc<float>(&p_in.weight, N * sizeof(float));
  
  cudaMalloc<float>(&p_out.pos_x, N * sizeof(float));
  cudaMalloc<float>(&p_out.pos_y, N * sizeof(float));
  cudaMalloc<float>(&p_out.pos_z, N * sizeof(float));
  cudaMalloc<float>(&p_out.vel_x, N * sizeof(float));
  cudaMalloc<float>(&p_out.vel_y, N * sizeof(float));
  cudaMalloc<float>(&p_out.vel_z, N * sizeof(float));
  p_out.weight = p_in.weight;

  cudaMalloc<float>(&comX, sizeof(float));
  cudaMalloc<float>(&comY, sizeof(float));
  cudaMalloc<float>(&comZ, sizeof(float));
  cudaMalloc<float>(&comW, sizeof(float));
  cudaMalloc<int>(&lock, sizeof(float));
 
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                       FILL IN: memory transfers (step 0)                                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  cudaMemcpy(p_in.pos_x, particles_cpu.pos_x, N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(p_in.pos_y, particles_cpu.pos_y, N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(p_in.pos_z, particles_cpu.pos_z, N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(p_in.vel_x, particles_cpu.vel_x, N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(p_in.vel_y, particles_cpu.vel_y, N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(p_in.vel_z, particles_cpu.vel_z, N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(p_in.weight, particles_cpu.weight, N * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 blockDim(thr_blc);
  dim3 gridDim(simulationGrid);
  
  gettimeofday(&t1, 0);

  for(int s = 0; s < steps; s++)
  {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: kernels invocation (step 0)                                     //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cudaMemsetAsync(comX, 0, sizeof(float), stream1);
    cudaMemsetAsync(comY, 0, sizeof(float), stream1);
    cudaMemsetAsync(comZ, 0, sizeof(float), stream1);
    cudaMemsetAsync(comW, 0, sizeof(float), stream1);
    cudaMemsetAsync(lock, 0, sizeof(int), stream1);

    calculate_velocity<<<gridDim, blockDim,  (7 * thr_blc + 6) * sizeof(float), stream0>>>(p_out, p_in, N, dt);
    centerOfMass<<<reductionGrid, red_thr_blc, 4 * red_thr_blc * sizeof(float), stream1>>>(p_in, comX, comY, comZ, comW, lock, N);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                          FILL IN: synchronization  (step 4)                                    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (writeFreq > 0 && (s % writeFreq == 0))
    {
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //                          FILL IN: synchronization and file access logic (step 4)                             //
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      cudaMemcpyAsync(particles_cpu.pos_x, p_in.pos_x, N * sizeof(float),
              cudaMemcpyDeviceToHost, stream2);
      cudaMemcpyAsync(particles_cpu.pos_y, p_in.pos_y, N * sizeof(float),
              cudaMemcpyDeviceToHost, stream2);
      cudaMemcpyAsync(particles_cpu.pos_z, p_in.pos_z, N * sizeof(float),
              cudaMemcpyDeviceToHost, stream2);
      cudaMemcpyAsync(particles_cpu.vel_x, p_in.vel_x, N * sizeof(float),
              cudaMemcpyDeviceToHost, stream2);
      cudaMemcpyAsync(particles_cpu.vel_y, p_in.vel_y, N * sizeof(float),
              cudaMemcpyDeviceToHost, stream2);
      cudaMemcpyAsync(particles_cpu.vel_z, p_in.vel_z, N * sizeof(float),
              cudaMemcpyDeviceToHost, stream2);

      cudaMemcpyAsync(&comOnGPUA->x, comX, sizeof(float),
              cudaMemcpyDeviceToHost, stream1);
      cudaMemcpyAsync(&comOnGPUA->y, comY, sizeof(float),
              cudaMemcpyDeviceToHost, stream1);
      cudaMemcpyAsync(&comOnGPUA->z, comZ, sizeof(float),
              cudaMemcpyDeviceToHost, stream1);
      cudaMemcpyAsync(&comOnGPUA->w, comW, sizeof(float),
              cudaMemcpyDeviceToHost, stream1);

      cudaStreamSynchronize(stream2);
      h5Helper.writeParticleData(s / writeFreq);

      cudaStreamSynchronize(stream1);
      h5Helper.writeCom(comOnGPUA->x, comOnGPUA->y, comOnGPUA->z, comOnGPUA->w, s / writeFreq);
    }
    
    std::swap(p_in, p_out);
    cudaDeviceSynchronize();
  }


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                           //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  

  cudaMemset(comX, 0, sizeof(float));
  cudaMemset(comY, 0, sizeof(float));
  cudaMemset(comZ, 0, sizeof(float));
  cudaMemset(comW, 0, sizeof(float));
  cudaMemset(lock, 0, sizeof(int));
  
  centerOfMass<<<reductionGrid, red_thr_blc, 4 * red_thr_blc * sizeof(float)>>>(p_in, comX, comY, comZ, comW, lock, N);

  cudaDeviceSynchronize();

  gettimeofday(&t2, 0);

  // Approximate simulation wall time
  double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
  printf("Time: %f s\n", t);


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                             FILL IN: memory transfers for particle data (step 0)                                 //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cudaMemcpy(particles_cpu.pos_x, p_in.pos_x, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.pos_y, p_in.pos_y, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.pos_z, p_in.pos_z, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_x, p_in.vel_x, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_y, p_in.vel_y, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(particles_cpu.vel_z, p_in.vel_z, N * sizeof(float),
             cudaMemcpyDeviceToHost);
  
  float4 comOnGPU;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cudaMemcpy(&comOnGPU.x, comX, sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&comOnGPU.y, comY, sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&comOnGPU.z, comZ, sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&comOnGPU.w, comW, sizeof(float),
             cudaMemcpyDeviceToHost);
  
  float4 comOnCPU = centerOfMassCPU(md);

  std::cout << "Center of mass on CPU:" << std::endl
            << comOnCPU.x <<", "
            << comOnCPU.y <<", "
            << comOnCPU.z <<", "
            << comOnCPU.w
            << std::endl;

  std::cout << "Center of mass on GPU:" << std::endl
            << comOnGPU.x<<", "
            << comOnGPU.y<<", "
            << comOnGPU.z<<", "
            << comOnGPU.w
            << std::endl;

  // Writing final values to the file
  h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
  h5Helper.writeParticleDataFinal();

  return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
