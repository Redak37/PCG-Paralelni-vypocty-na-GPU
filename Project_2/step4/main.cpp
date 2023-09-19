/**
 * @file      main.cpp
 *
 * @author    Radek Duchon
 *            Faculty of Information Technology
 *            Brno University of Technology
 *            xducho07@stud.fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2021
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      15 November  2021, 14:03 (revised) \n
 *
 */

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main routine of the project
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
  // Parse command line parameters
  if (argc != 7)
  {
    printf("Usage: nbody <N> <dt> <steps> <write intesity> <input> <output>\n");
    exit(EXIT_FAILURE);
  }

  const int   N         = std::stoi(argv[1]);
  const float dt        = std::stof(argv[2]);
  const int   steps     = std::stoi(argv[3]);
  const int   writeFreq = (std::stoi(argv[4]) > 0) ? std::stoi(argv[4]) : 0;

  printf("N: %d\n", N);
  printf("dt: %f\n", dt);
  printf("steps: %d\n", steps);

  const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                         Code to be implemented                                                   //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // 1.  Memory allocation on CPU
  Particles p_in(N);
  Particles p_out(N);
  float4 *comOnGPU = (float4 *)malloc(sizeof(*comOnGPU));
  #pragma acc enter data create(comOnGPU[0])


  // 2. Create memory descriptor
  /*
   * Caution! Create only after CPU side allocation
   * parameters:
   *                                    Stride of two               Offset of the first
   *             Data pointer           consecutive elements        element in floats,
   *                                    in floats, not bytes        not bytes
  */
  MemDesc md(
        p_in.pos_x,        1,                      0,              // Postition in X
        p_in.pos_y,        1,                      0,              // Postition in Y
        p_in.pos_z,        1,                      0,              // Postition in Z
        p_in.vel_x,        1,                      0,              // Velocity in X
        p_in.vel_y,        1,                      0,              // Velocity in Y
        p_in.vel_z,        1,                      0,              // Velocity in Z
        p_in.weight,       1,                      0,              // Weight
        N,                                                         // Number of p_in
        recordsNum);                                               // Number of records in output file



  H5Helper h5Helper(argv[5], argv[6], md);

  // Read data
  try
  {
    h5Helper.init();
    h5Helper.readParticleData();
  } catch (const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return EXIT_FAILURE;
  }


  std::memcpy(p_out.weight, p_in.weight, N * sizeof(*p_out.weight));
  #pragma acc update device(p_out.weight[N])
  // 3. Copy data to GPU
  p_in.copyToGPU(); 

  // Start the time
  auto startTime = std::chrono::high_resolution_clock::now();



  // 4. Run the loop - calculate new Particle positions.
  for (int s = 0; s < steps; s++)
  {
    /// In step 4 - fill in the code to store Particle snapshots.
    if (writeFreq > 0 && (s % writeFreq == 0))
    {
      centerOfMassGPU(comOnGPU, (s & 1) ? p_out : p_in, N);

      if (s > 0) {
        if (s & 1)
          p_out.copyToCPUAsync();
        else
          p_in.copyToCPUAsync();
      }

      #pragma acc wait(UPDATE_PARTICLES)
      if (s & 1) {
        std::memcpy(p_in.pos_x, p_out.pos_x, N * sizeof(float));
        std::memcpy(p_in.pos_y, p_out.pos_y, N * sizeof(float));
        std::memcpy(p_in.pos_z, p_out.pos_z, N * sizeof(float));
        std::memcpy(p_in.vel_x, p_out.vel_x, N * sizeof(float));
        std::memcpy(p_in.vel_x, p_out.vel_y, N * sizeof(float));
        std::memcpy(p_in.vel_z, p_out.vel_z, N * sizeof(float));
      }

      h5Helper.writeParticleData(s / writeFreq);

      #pragma acc update host(comOnGPU[0]) wait(UPDATE_COM) async(COMPUTE_COM)
      #pragma acc wait(COMPUTE_COM)
      h5Helper.writeCom(comOnGPU->x, comOnGPU->y, comOnGPU->z, comOnGPU->w, s / writeFreq);
    }

    calculate_velocity((s & 1) ? p_in : p_out, (s & 1) ? p_out : p_in, N, dt);
  }// for s ...

  // 5. In steps 3 and 4 -  Compute center of gravity
  centerOfMassGPU(comOnGPU, (steps & 1) ? p_out : p_in, N);



  // Stop watchclock
  const auto   endTime = std::chrono::high_resolution_clock::now();
  const double time    = (endTime - startTime) / std::chrono::milliseconds(1);
  printf("Time: %f s\n", time / 1000);


  // 5. Copy data from GPU back to CPU.
  if (steps & 1) {
      p_out.copyToCPU();
      std::memcpy(p_in.pos_x, p_out.pos_x, N * sizeof(float));
      std::memcpy(p_in.pos_y, p_out.pos_y, N * sizeof(float));
      std::memcpy(p_in.pos_z, p_out.pos_z, N * sizeof(float));
      std::memcpy(p_in.vel_x, p_out.vel_x, N * sizeof(float));
      std::memcpy(p_in.vel_x, p_out.vel_y, N * sizeof(float));
      std::memcpy(p_in.vel_z, p_out.vel_z, N * sizeof(float));
  } else
      p_in.copyToCPU();



  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /// Calculate center of gravity
  float4 comOnCPU = centerOfMassCPU(md);


    std::cout<<"Center of mass on CPU:"<<std::endl
      << comOnCPU.x <<", "
      << comOnCPU.y <<", "
      << comOnCPU.z <<", "
      << comOnCPU.w
      << std::endl;

    #pragma acc update host(comOnGPU[0]) wait(UPDATE_COM)
    std::cout<<"Center of mass on GPU:"<<std::endl
      << comOnGPU->x <<", "
      << comOnGPU->y <<", "
      << comOnGPU->z <<", "
      << comOnGPU->w
      <<std::endl;

  // Store final positions of the particles into a file
  h5Helper.writeComFinal(comOnGPU->x, comOnGPU->y, comOnGPU->z, comOnGPU->w);
  h5Helper.writeParticleDataFinal();

  #pragma acc exit data delete(comOnGPU[0])
  std::free(comOnGPU);

  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

