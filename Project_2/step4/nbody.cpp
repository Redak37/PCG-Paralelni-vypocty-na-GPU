/**
 * @file      nbody.cpp
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

#include <math.h>
#include <cfloat>
#include "nbody.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Calculate velocity
void calculate_velocity(const Particles& p_out,
                        const Particles& p_in,
                        const int        N,
                        const float      dt)
{
    #pragma acc parallel loop present(p_in, p_out) gang worker vector async(COMPUTE_VELOCITIES)
    for (unsigned int i = 0; i < N; ++i) {
        const float pos_x = p_in.pos_x[i];
        const float pos_y = p_in.pos_y[i];
        const float pos_z = p_in.pos_z[i];
        const float vel_x = p_in.vel_x[i];
        const float vel_y = p_in.vel_y[i];
        const float vel_z = p_in.vel_z[i];
        const float weight = p_in.weight[i];

        float r2, dx, dy, dz;
        float vx = 0.f, vy = 0.f, vz = 0.f;

        #pragma acc loop seq
        for (int j = 0; j < N; j++) {
            dx = pos_x - p_in.pos_x[j];
            dy = pos_y - p_in.pos_y[j];
            dz = pos_z - p_in.pos_z[j];

            r2 = dx*dx + dy*dy + dz*dz;
            const float weight2 = p_in.weight[j];

            if (r2 > COLLISION_DISTANCE * COLLISION_DISTANCE) {
                const float Fg_dt_m2_r = (-G * dt / (sqrtf(r2) * r2)) * weight2;

                vx += Fg_dt_m2_r * dx;
                vy += Fg_dt_m2_r * dy;
                vz += Fg_dt_m2_r * dz;
            } else if (r2 > 0.f) {
                const float tmp = 1.f / (weight + weight2);
                vx += (((weight - weight2) * vel_x + 2 * weight2 * p_in.vel_x[j]) *
                       tmp) - vel_x ;
                vy += (((weight - weight2) * vel_y + 2 * weight2 * p_in.vel_y[j]) *
                       tmp) - vel_y ;
                vz += (((weight - weight2) * vel_z + 2 * weight2 * p_in.vel_z[j]) *
                       tmp) - vel_z ;
            }
        }

        vx += vel_x;
        vy += vel_y;
        vz += vel_z;

        p_out.vel_x[i] = vx;
        p_out.vel_y[i] = vy;
        p_out.vel_z[i] = vz;

        p_out.pos_x[i] = pos_x + vx * dt;
        p_out.pos_y[i] = pos_y + vy * dt;
        p_out.pos_z[i] = pos_z + vz * dt;
    }
}



/// Compute center of gravity
void centerOfMassGPU(float4 *com,
                     const Particles& p,
                     const int        N)
{
    float w = 0.f;
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
   
    #pragma acc parallel loop present(p) reduction(+:x, y, z, w) gang worker vector\
        wait(COMPUTE_VELOCITIES) async(COMPUTE_COM)
    for (unsigned i = 0; i < N; ++i) {
        const float weight = p.weight[i];
        w += weight;
        x += weight * p.pos_x[i];
        y += weight * p.pos_y[i];
        z += weight * p.pos_z[i];
    }

    #pragma acc wait(COMPUTE_COM) async(UPDATE_COM)
    *com = {x / w, y / w, z / w, w};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
