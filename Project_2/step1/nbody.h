/**
 * @file      nbody.h
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
 * @date      15 November  2021, 14:10 (revised) \n
 *
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include  <cmath>
#include "h5Helper.h"

/// Gravity constant
constexpr float G = 6.67384e-11f;

/// Collision distance threshold
constexpr float COLLISION_DISTANCE = 0.01f;

/**
 * @struct float4
 * Structure that mimics CUDA float4
 */
struct float4
{
  float x;
  float y;
  float z;
  float w;
};

/// Define sqrtf from CUDA libm library
#pragma acc routine(sqrtf) seq

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Structure with particle data
 */
struct Particles
{
  // Fill the structure holding the particle/s data
  // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines
  size_t n;
  float *pos_x;
  float *pos_y;
  float *pos_z;
  float *vel_x;
  float *vel_y;
  float *vel_z;
  float *weight;

  // Default constructor is not allowed - the size must be known in advance
  Particles() = delete;
  // Copy constructor is not allows - performance loss due to data copies.
  Particles(const Particles& stc) = delete;
  // Assignment operator is not allowed - performance loss due to data copies.
  Particles& operator=(const Particles& stc) = delete;

  Particles(size_t n) : n(n)
  {
    pos_x = new float[n];
    pos_y = new float[n];
    pos_z = new float[n];
    vel_x = new float[n];
    vel_y = new float[n];
    vel_z = new float[n];
    weight = new float[n];

    #pragma acc enter data copyin(this)
    #pragma acc enter data create(pos_x[n])
    #pragma acc enter data create(pos_y[n])
    #pragma acc enter data create(pos_z[n])
    #pragma acc enter data create(vel_x[n])
    #pragma acc enter data create(vel_y[n])
    #pragma acc enter data create(vel_z[n])
    #pragma acc enter data create(weight[n])
  }

  ~Particles()
  {
    #pragma acc exit data delete(pos_x[n])
    #pragma acc exit data delete(pos_y[n])
    #pragma acc exit data delete(pos_z[n])
    #pragma acc exit data delete(vel_x[n])
    #pragma acc exit data delete(vel_y[n])
    #pragma acc exit data delete(vel_z[n])
    #pragma acc exit data delete(weight[n])
    #pragma acc exit data delete(this)

    delete[] pos_x;
    delete[] pos_y;
    delete[] pos_z;
    delete[] vel_x;
    delete[] vel_y;
    delete[] vel_z;
    delete[] weight;
  }

  void copyToGPU()
  {
    #pragma acc update device(pos_x[n])
    #pragma acc update device(pos_y[n])
    #pragma acc update device(pos_z[n])
    #pragma acc update device(vel_x[n])
    #pragma acc update device(vel_y[n])
    #pragma acc update device(vel_z[n])
    #pragma acc update device(weight[n])
  }
  
  void copyToCPU()
  {
    #pragma acc update host(pos_x[n])
    #pragma acc update host(pos_y[n])
    #pragma acc update host(pos_z[n])
    #pragma acc update host(vel_x[n])
    #pragma acc update host(vel_y[n])
    #pragma acc update host(vel_z[n])
  }
};// end of Particles
//----------------------------------------------------------------------------------------------------------------------

/**
 * @struct Velocities
 * Velocities of the particles
 */
struct Velocities
{
  // Fill the structure holding the particle/s data
  // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines
  size_t n;
  float *vel_x;
  float *vel_y;
  float *vel_z;

  // Default constructor is not allowed - the size must be known in advance
  Velocities() = delete;
  // Copy constructor is not allows - performance loss due to data copies.
  Velocities(const Velocities& stc) = delete;
  // Assignment operator is not allowed - performance loss due to data copies.
  Velocities& operator=(const Velocities& stc) = delete;

  Velocities(size_t n) : n(n)
  {
    vel_x = new float[n];
    vel_y = new float[n];
    vel_z = new float[n];

    #pragma acc enter data copyin(this)
    #pragma acc enter data create(vel_x[n])
    #pragma acc enter data create(vel_y[n])
    #pragma acc enter data create(vel_z[n])
  }

  ~Velocities()
  {
    #pragma acc exit data delete(vel_x[n])
    #pragma acc exit data delete(vel_y[n])
    #pragma acc exit data delete(vel_z[n])
    #pragma acc exit data delete(this)

    delete[] vel_x;
    delete[] vel_y;
    delete[] vel_z;
  }

  void copyToGPU()
  {
    #pragma acc update device(vel_x[n])
    #pragma acc update device(vel_y[n])
    #pragma acc update device(vel_z[n])
  }
  
  void copyToCPU()
  {
    #pragma acc update host(vel_x[n])
    #pragma acc update host(vel_y[n])
    #pragma acc update host(vel_z[n])
  }
};// end of Velocities
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute gravitation velocity
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void calculate_gravitation_velocity(const Particles& p,
                                    Velocities&      tmp_vel,
                                    const int        N,
                                    const float      dt);

/**
 * Calculate collision velocity
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void calculate_collision_velocity(const Particles& p,
                                  Velocities&      tmp_vel,
                                  const int        N,
                                  const float      dt);

/**
 * Update particle position
 * @param [in]  p        - Particles
 * @param [out] tmp_vel  - Temporal velocity
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void update_particle(const Particles& p,
                     Velocities&      tmp_vel,
                     const int        N,
                     const float      dt);



/**
 * Compute center of gravity - implement in steps 3 and 4.
 * @param [in] p - Particles
 * @param [in] N - Number of particles
 * @return Center of Mass [x, y, z] and total weight[w]
 */
float4 centerOfMassGPU(const Particles& p,
                       const int        N);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Compute center of mass on CPU
 * @param memDesc
 * @return centre of gravity
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */
