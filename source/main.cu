#include <iostream>
#include <string>
#include "../include/vec.cuh"
#include "../include/constants.h"
#include "../include/particle.cuh"
#include "../include/vtkwrite.h"
#include "../include/vtkgrid.h"


class Quaternion {
public:
    double w, x, y, z;

    __host__ __device__ Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}

    __host__ __device__ Quaternion(double ww, double xx, double yy, double zz) : w(ww), x(xx), y(yy), z(zz) {}

    __host__ __device__ void normalize() {
        double norm = sqrt(w * w + x * x + y * y + z * z);
        w = w / norm; x = x / norm; y = y / norm; z = z / norm;
    }

    __host__ __device__ Vector3d rotate(const Vector3d& v) const {
        Quaternion p(0, v.x, v.y, v.z);
        Quaternion q = (*this) * p * inverse();
        return Vector3d(q.x, q.y, q.z);
    }

    __host__ __device__ Quaternion inverse() const {
        double normSq = w * w + x * x + y * y + z * z;
        return Quaternion(w / normSq, -x / normSq, -y / normSq, -z / normSq);
    }

    // Quaternion multiplication
    __host__ __device__ Quaternion operator*(const Quaternion& q) const {
        double ww = w * q.w - x * q.x - y * q.y - z * q.z;
        double xx = w * q.x + x * q.w + y * q.z - z * q.y;
        double yy = w * q.y - x * q.z + y * q.w + z * q.x;
        double zz = w * q.z + x * q.y - y * q.x + z * q.w;
        return Quaternion(ww, xx, yy, zz);
    }

    __host__ __device__ Quaternion operator*(const double& d) const {
        return Quaternion(d * w, d * x, d * y, z + d * z);
    }

    __host__ __device__ Quaternion operator+(const Quaternion& q) const {
        return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
    }
};

__global__ void compute_position(Particle* particles, size_t n, double dt,Const c_h, Quaternion* rotations){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < n){ //for every particle

        //Position x(t+dt)
        particles[i].pos = particles[i].pos + particles[i].vel * dt;
        //velocity intermediate v(t+dt/2)
        rotations[i] = rotations[i] + Quaternion(0, particles[i].ang_vel.x, particles[i].ang_vel.y, particles[i].ang_vel.z) * rotations[i] * 0.5 * dt;
        rotations[i].normalize();

        // Apply quaternion rotation
        particles[i].ang_pos = rotations[i].rotate(particles[i].ang_pos);

        // particles[i].pos = q.rotate(particles[i].pos);
        // particles[i].vel = q.rotate(particles[i].vel);

        if(particles[i].pos.z <= -1 || particles[i].pos.z >= 3)
            {particles[i].vel.z = -particles[i].vel.z;}
        
        if(particles[i].pos.x <= -1 || particles[i].pos.x >= 3)
           { particles[i].vel.x = -particles[i].vel.x;}
        
        if(particles[i].pos.y <= -1 || particles[i].pos.y >= 3)
           { particles[i].vel.y = -particles[i].vel.y;}
    }
}

//Heavyside function used in granular force computation
__host__ __device__ bool step(double x){
    return (x < 0) ? 0 : 1;
}

// -------------------New CODE----------------By Chirag----------------//
__global__ void compute_acceleration_sd(Particle* particles, size_t n, double dt, Vector3d g, Const c_h){
    size_t i =  blockDim.x * blockIdx.x + threadIdx.x;

    if(i < n){ //acceleration a(t+dt)

        Vector3d torque, force;
        force = force + c_h.M * g;

        for(size_t j = 0; j < n; j++){
            

            if (i==j){
                continue;
            }

            __syncthreads();
            
            Vector3d x_ij = particles[i].pos - particles[j].pos;
            double x_norm = x_ij.norm();
            
            if (x_norm < particles[i].radius + particles[j].radius){
                
                Vector3d force_t;
                Vector3d force_n;

                Vector3d v_ij = particles[i].vel - particles[j].vel;
                Vector3d x_unit = x_ij / x_norm;
                Vector3d r_ij = x_ij - x_unit * particles[j].radius;

                Vector3d v_ij_s = v_ij - x_unit * dot(x_unit, v_ij) +
                    cross(x_unit, (particles[i].radius * particles[i].ang_vel + particles[j].radius * particles[j].ang_vel));

                force_n = c_h.K * x_unit * (c_h.sigma - x_norm) * step(c_h.sigma - x_norm) - 
                    c_h.gamma * x_unit * dot(x_unit, (particles[i].vel - particles[j].vel)) * step(c_h.sigma - x_norm); //Granular forces(Spring-Dashpot) 

                force_t =  -1 * fminf(c_h.gamma_t * v_ij_s.norm(), c_h.mu * force_n.norm()) * v_ij_s;          

                force = force + force_n + force_t; 

                torque = torque - particles[i].radius * cross(r_ij, force_t);
            }
            __syncthreads();
        }
        particles[i].acc = force/ c_h.M;
        particles[i].ang_acc = torque / particles[i].moi; 
    }
}

__global__ void compute_velocity(Particle* particles, size_t n, double dt){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        //velocity v(t+dt)
        particles[i].vel = particles[i].vel + particles[i].acc * dt;

        particles[i].ang_vel = particles[i].ang_vel + particles[i].ang_acc * dt;
    }
}


int main(int argc, char* argv[]){
    //command line arguments
    size_t n = std::stoi(argv[1]);
    size_t CR = ceil(cbrtf(n));
    double T = std::stod(argv[2]);
    double dt = std::stod(argv[3]);
    double gx = std::stod(argv[4]);
    double gy = std::stod(argv[5]);
    double gz = std::stod(argv[6]);
    Vector3d g(gx, gy, gz);

    //remove previous .vtk files
    system("rm -r ../output/*.vtk");

    //Create constants
    const Const c_h("../constants.txt");

    ////Initialize random seed
    // std::random_device rd;
    // std::mt19937 gen(rd());
    
    //Dynamically allocate n particles
    Particle* particles = new Particle[n];
    Particle* particles_device; 

    for(size_t i = 0; i < n; i++){
        // Initialize the particles id
        particles[i] = Particle(i + 1);
        // particles[i].initialize_grid(i);
    }

    //read particle data from file
    initialize_from_file(particles, "../initial_conditions.txt");

    // Create bounding box
    create_wireframe(-1.5, 3.5, -1.5, 3.5, -1.5, 3.5);

    //copy particles to gpu
    size_t size = n * sizeof(Particle);
    //allocate and copy particles to gpu
    cudaMalloc(&particles_device, size);
    cudaMemcpy(particles_device, particles, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    int iter = 0;

    Quaternion* rotations = new Quaternion[n];
    Quaternion* rotations_device;

    // Initialize the rotations to identity quaternions
    for(size_t i = 0; i < n; i++){
        rotations[i] = Quaternion(1, 0, 0, 0);
    }

    // Copy rotations to the GPU
    size_t rotationsSize = n * sizeof(Quaternion);
    cudaMalloc(&rotations_device, rotationsSize);
    cudaMemcpy(rotations_device, rotations, rotationsSize, cudaMemcpyHostToDevice);


    for(double t = 0; t<=T; t += dt){
        // std::cout<<"time:"<<t<<" "<<std::endl;

        compute_position<<<blocksPerGrid, threadsPerBlock>>>(particles_device, n, dt,c_h,rotations_device);
        cudaDeviceSynchronize();

        compute_acceleration_sd<<<blocksPerGrid, threadsPerBlock>>>(particles_device, n, dt, g, c_h);
        cudaDeviceSynchronize();

        compute_velocity<<<blocksPerGrid, threadsPerBlock>>>(particles_device, n, dt);
        cudaDeviceSynchronize();
        
        cudaMemcpy(particles, particles_device, size, cudaMemcpyDeviceToHost);
    
        if(iter%5000 == 0)            
            write_vtk(particles, n, iter);
        iter++;
    }
    cudaFree(particles_device);
    delete[] particles;

    cudaFree(rotations_device);
    delete[] rotations;
}
