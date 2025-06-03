__global__ void calculate_drift_without_noise(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, cuDoubleComplex *kin, cuDoubleComplex *kin_conjug,cuDoubleComplex *drift, cuDoubleComplex *drift_conjug, double factor, double P, double MU)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);

	cuDoubleComplex tempm_0=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempp_0=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempm_1=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempp_1=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempm_2=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempp_2=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempm_3=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempp_3=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex term_p= make_cuDoubleComplex(0.,0.);
	cuDoubleComplex term_i= make_cuDoubleComplex(0.,0.);
	cuDoubleComplex term_param= make_cuDoubleComplex(0.,0.);
	cuDoubleComplex term_inter= make_cuDoubleComplex(0.,0.);

	cuDoubleComplex N_11, N_12, N_21, N_22;
	cuDoubleComplex N_11_conjug, N_12_conjug, N_21_conjug, N_22_conjug;

	N_11 = psi[ind(0,j,k,l,m-1)] *psi_conjug[ind(0,j,k,l,m)];
	N_12 = psi[ind(1,j,k,l,m-1)] *psi_conjug[ind(1,j,k,l,m)];
	N_21 = psi[ind(2,j,k,l,m-1)] *psi_conjug[ind(2,j,k,l,m)];
	N_22 = psi[ind(3,j,k,l,m-1)] *psi_conjug[ind(3,j,k,l,m)];

	N_11_conjug = psi[ind(0,j,k,l,m)] *psi_conjug[ind(0,j,k,l,m+1)];
	N_12_conjug = psi[ind(1,j,k,l,m)] *psi_conjug[ind(1,j,k,l,m+1)];
	N_21_conjug = psi[ind(2,j,k,l,m)] *psi_conjug[ind(2,j,k,l,m+1)];
	N_22_conjug = psi[ind(3,j,k,l,m)] *psi_conjug[ind(3,j,k,l,m+1)];

	// define psi_0 = psi_11, psi_1 = psi_12, psi_2 = psi_21, psi_3 = psi_22
	tempm_0 = (N_11 + N_12 + N_21) *psi[ind(0,j,k,l,m-1)]
				+(psi[ind(1,j,k,l,m-1)] *psi_conjug[ind(3,j,k,l,m)])*psi[ind(2,j,k,l,m-1)];

	tempm_1 = (N_11 + N_12 + N_22)*psi[ind(1,j,k,l,m-1)]
				+(psi[ind(0,j,k,l,m-1)] *psi_conjug[ind(2,j,k,l,m)]) *psi[ind(3,j,k,l,m-1)];

	tempm_2 = (N_11 + N_21 + N_22) *psi[ind(2,j,k,l,m-1)]
				+(psi[ind(3,j,k,l,m-1)] *psi_conjug[ind(1,j,k,l,m)]) *psi[ind(0,j,k,l,m-1)];

	tempm_3 = (N_12 + N_21 + N_22) *psi[ind(3,j,k,l,m-1)]
				+(psi[ind(2,j,k,l,m-1)] *psi_conjug[ind(0,j,k,l,m)]) *psi[ind(1,j,k,l,m-1)];

	tempp_0 = (N_11_conjug + N_12_conjug + N_21_conjug) *psi_conjug[ind(0,j,k,l,m+1)]
				+psi_conjug[ind(1,j,k,l,m+1)] *psi[ind(3,j,k,l,m)] *psi_conjug[ind(2,j,k,l,m+1)];

	tempp_1 = (N_11_conjug + N_12_conjug + N_22_conjug) *psi_conjug[ind(1,j,k,l,m+1)]
				+psi_conjug[ind(0,j,k,l,m+1)] *psi[ind(2,j,k,l,m)] *psi_conjug[ind(3,j,k,l,m+1)];//hier auch ein conjug?

	tempp_2 = (N_11_conjug + N_21_conjug + N_22_conjug) *psi_conjug[ind(2,j,k,l,m+1)]
				+psi_conjug[ind(3,j,k,l,m+1)] *psi[ind(1,j,k,l,m)] *psi_conjug[ind(0,j,k,l,m+1)];

	tempp_3 = (N_12_conjug + N_21_conjug + N_22_conjug) *psi_conjug[ind(3,j,k,l,m+1)]
				+psi_conjug[ind(2,j,k,l,m+1)] *psi[ind(0,j,k,l,m)] *psi_conjug[ind(1,j,k,l,m+1)];

	
	double V=0;

		// i = 0
		{
			cuDoubleComplex term_param = term_p + (MU - V + P + Q) * psi[ind(0,j,k,l,m-1)];
			cuDoubleComplex term_inter = term_i - G * tempm_0;
			drift[ind(0,j,k,l,m)] = factor * a_x * (
				psi[ind(0,j,k,l,m-1)] - psi[ind(0,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin[ind(0,j,k,l,m-1)])
			);

			term_param = term_p + (MU - V + P + Q) * psi[ind(0,j,k,l,m+1)];
			term_inter = term_i - G * tempp_0;
			drift_conjug[ind(0,j,k,l,m)] = factor * a_x * (
				psi_conjug[ind(0,j,k,l,m+1)] - psi_conjug[ind(0,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin_conjug[ind(0,j,k,l,m+1)])
			);
		}

		// i = 1
		{
			cuDoubleComplex term_param = term_p + (MU - V) * psi[ind(1,j,k,l,m-1)];
			cuDoubleComplex term_inter = term_i - G * tempm_1;
			drift[ind(1,j,k,l,m)] = factor * a_x * (
				psi[ind(1,j,k,l,m-1)] - psi[ind(1,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin[ind(1,j,k,l,m-1)])
			);

			term_param = term_p + (MU - V) * psi[ind(1,j,k,l,m+1)];
			term_inter = term_i - G * tempp_1;
			drift_conjug[ind(1,j,k,l,m)] = factor * a_x * (
				psi_conjug[ind(1,j,k,l,m+1)] - psi_conjug[ind(1,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin_conjug[ind(1,j,k,l,m+1)])
			);
		}

		// i = 2
		{
			cuDoubleComplex term_param = term_p + (MU - V) * psi[ind(2,j,k,l,m-1)];
			cuDoubleComplex term_inter = term_i - G * tempm_2;
			drift[ind(2,j,k,l,m)] = factor * a_x * (
				psi[ind(2,j,k,l,m-1)] - psi[ind(2,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin[ind(2,j,k,l,m-1)])
			);

			term_param = term_p + (MU - V) * psi[ind(2,j,k,l,m+1)];
			term_inter = term_i - G * tempp_2;
			drift_conjug[ind(2,j,k,l,m)] = factor * a_x * (
				psi_conjug[ind(2,j,k,l,m+1)] - psi_conjug[ind(2,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin_conjug[ind(2,j,k,l,m+1)])
			);
		}

		// i = 3
		{
			cuDoubleComplex term_param = term_p + (MU - V - P + Q) * psi[ind(3,j,k,l,m-1)];
			cuDoubleComplex term_inter = term_i - G * tempm_3;
			drift[ind(3,j,k,l,m)] = factor * a_x * (
				psi[ind(3,j,k,l,m-1)] - psi[ind(3,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin[ind(3,j,k,l,m-1)])
			);

			term_param = term_p + (MU - V - P + Q) * psi[ind(3,j,k,l,m+1)];
			term_inter = term_i - G * tempp_3;
			drift_conjug[ind(3,j,k,l,m)] = factor * a_x * (
				psi_conjug[ind(3,j,k,l,m+1)] - psi_conjug[ind(3,j,k,l,m)]
				+ EPS * (term_param + term_inter + kin_conjug[ind(3,j,k,l,m+1)])
			);
		}
}

__global__ void multiply_by_timestep(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, double dt)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	for(int i=0; i<COMPONENTS; i++)
	{
		psi[ind(i,j,k,l,m)]*=dt;
		psi_conjug[ind(i,j,k,l,m)]*=dt;
	}
}

__global__ void add_noise(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, double *random)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	for(int i=0; i<COMPONENTS; i++)
	{
		psi[ind(i,j,k,l,m)]+=make_cuDoubleComplex(random[ind_rand(i,j,k,l,m,0)],random[ind_rand(i,j,k,l,m,1)]);
		psi_conjug[ind(i,j,k,l,m)]+=make_cuDoubleComplex(random[ind_rand(i,j,k,l,m,0)],-random[ind_rand(i,j,k,l,m,1)]);
	}
}


__global__ void compute_kinetic_part(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug,cuDoubleComplex *result, cuDoubleComplex *result_conjug)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double px,py,pz;
	
	if(j<XSIZE/2||XSIZE==1)
	{
		px=2.*M_PI*double(j)/double(XSIZE);
	}
	else
	{
		px=-2.*M_PI*double(XSIZE-j)/double(XSIZE);
	}
	
	if(k<YSIZE/2||YSIZE==1)
	{
		py=2.*M_PI*double(k)/double(YSIZE);
	}
	else
	{
		py=-2.*M_PI*double(YSIZE-k)/double(YSIZE);
	}
			
	if(l<ZSIZE/2||ZSIZE==1)
	{
		pz=2.*M_PI*double(l)/double(ZSIZE);
	}
	else
	{
		pz=-2.*M_PI*double(ZSIZE-l)/double(ZSIZE);		
	}
	
	double p2=px*px; //+py*py+pz*pz
	
	for(int i=0; i<COMPONENTS; i++)
	{		
		result[ind(i,j,k,l,m)]=-p2*psi[ind(i,j,k,l,m)];
		result_conjug[ind(i,j,k,l,m)]=-p2*psi_conjug[ind(i,j,k,l,m)];
	}
}

__global__ void create_density(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, cuDoubleComplex *density)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	for(int i=0; i<COMPONENTS; i++)
	{
		density[ind(i,j,k,l,m)]=(psi_conjug[ind(i,j,k,l,m+1)]*psi[ind(i,j,k,l,m)])*a_x;
	}
}

// Hilfsfunktion für atomare Max-Operation auf double
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel zur Berechnung von maximalem und mittlerem Drift
__global__ void compute_max_and_mean_drift(
    cuDoubleComplex *drift, int total_size, 
    double *max_result, double *mean_result)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_max = 0.0;
    double local_sum = 0.0;

    if (idx < total_size) {
        double val = cuCabs(drift[idx]);
        local_max = val;
        local_sum = val;
    }

    sdata[tid] = local_max;
    sdata[blockDim.x + tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s])
                sdata[tid] = sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxDouble(max_result, sdata[0]);
        atomicAdd(mean_result, sdata[blockDim.x]);
    }
}

class Propagator{
	public:
		Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug);
		Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug, unsigned long long seed, unsigned long long seed2);
		~Propagator();
		void propagate(double P, double MU);
		void propagate_without_noise(double factor, double P, double MU);
		void get_max_and_mean_drift(double &max_drift, double &mean_drift);
		void compute_new_dt(double mean_drift, double max_drift, double &dt_new);
	private:
		ComplexLattice *fields, *fields_conjug;
		ComplexLattice *drift, *drift_conjug;
		ComplexLattice *kin, *kin_conjug;
		double max_drift, mean_drift; dt_new;
		dim3 dimblock;
		dim3 dimgrid;
		curandGenerator_t gen;
		double *randomstore;	
};
	
Propagator::Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug)
{
	fields=lattice_fields;
	fields_conjug=lattice_fields_conjug;
	drift=lattice_drift;
	drift_conjug=lattice_drift_conjug;
	dt_new = DT;
	
	if((XSIZE%BLOCKSIZEX !=0)||
	   (YSIZE%BLOCKSIZEY !=0)||
	   (ZSIZE%BLOCKSIZEZ !=0)||
	   (TAUSIZE%BLOCKSIZETAU !=0))
	{
		cout<<"Length of the lattice dimensions must be a multiple of respective block size!"<<endl;
	}
	dim3 temp_dimblock(BLOCKSIZEX ,BLOCKSIZEY, BLOCKSIZEZ*BLOCKSIZETAU);
	dim3 temp_dimgrid(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE*TAUSIZE/(BLOCKSIZEZ*BLOCKSIZETAU)));
	dimblock=temp_dimblock;
	dimgrid=temp_dimgrid;
	
	kin=new ComplexLattice(false);
	kin_conjug=new ComplexLattice(true);
	
	cudaMalloc(&randomstore, 2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
}

Propagator::Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug, unsigned long long seed, unsigned long long seed2)
{
	fields=lattice_fields;
	fields_conjug=lattice_fields_conjug;
	drift=lattice_drift;
	drift_conjug=lattice_drift_conjug;
	dt_new = DT;
	
	if((XSIZE%BLOCKSIZEX !=0)||
	   (YSIZE%BLOCKSIZEY !=0)||
	   (ZSIZE%BLOCKSIZEZ !=0)||
	   (TAUSIZE%BLOCKSIZETAU !=0))
	{
		cout<<"Length of the lattice dimensions must be a multiple of respective block size!"<<endl;
	}
	dim3 temp_dimblock(BLOCKSIZEX ,BLOCKSIZEY, BLOCKSIZEZ*BLOCKSIZETAU);
	dim3 temp_dimgrid(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE*TAUSIZE/(BLOCKSIZEZ*BLOCKSIZETAU)));
	dimblock=temp_dimblock;
	dimgrid=temp_dimgrid;
	
	kin=new ComplexLattice(false);
	kin_conjug=new ComplexLattice(true);
	
	cudaMalloc(&randomstore, 2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
}

Propagator::~Propagator()
{
	cudaFree(randomstore);
	curandDestroyGenerator(gen);
}

void Propagator::get_max_and_mean_drift(double &max_drift, double &mean_drift)
{
    int total_size = COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE;
    double *d_max, *d_sum;
    cudaMalloc(&d_max, sizeof(double));
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_max, 0, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    int blockSize = dimblock.x * dimblock.y * dimblock.z;
    int gridSize = dimgrid.x * dimgrid.y * dimgrid.z;
    compute_max_and_mean_drift<<<gridSize, blockSize, 2*blockSize*sizeof(double)>>>(
        drift->get_pointer(), total_size, d_max, d_sum);

    double h_max, h_sum;
    cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    max_drift = h_max;
    mean_drift = h_sum / total_size;

    cudaFree(d_max);
    cudaFree(d_sum);
}

void Propagator::compute_new_dt(double mean_drift, double max_drift, double &dt_new)
{
    if (max_drift <= 0.0) {
        dt_new = DT;
    }
    else {
        dt_new = (mean_drift / max_drift) * DT;
    }

    double min_dt = 1e-8;
    double max_dt = 0.01;
    if (dt_new < min_dt) dt_new = min_dt;
    if (dt_new > max_dt) dt_new = max_dt;

    double alpha = 0.2;
    dt_new = (1 - alpha) * DT + alpha * dt_new;
}

void Propagator::propagate(double P, double MU)
{
	*kin=*fields;
	*kin_conjug=*fields_conjug;

	kin->fft();
	kin_conjug->fft();
	
	cudaDeviceSynchronize();
	compute_kinetic_part<<<dimgrid,dimblock>>>(kin->get_pointer(),kin_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer());
	cudaDeviceSynchronize();
	
	kin->fft_inv();
	kin_conjug->fft_inv();
	
	kin->normalize(1./double(a_x*XSIZE*YSIZE*ZSIZE));
	kin_conjug->normalize(1./double(a_x*XSIZE*YSIZE*ZSIZE));
	
	cudaDeviceSynchronize();
	calculate_drift_without_noise<<<dimgrid,dimblock>>>(fields->get_pointer(),fields_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer(),drift->get_pointer(),drift_conjug->get_pointer(), randomstore, randomstore, P, MU);
	cudaDeviceSynchronize();
	get_max_and_mean_drift(max_drift, mean_drift);
	cudaDeviceSynchronize();
	compute_new_dt(mean_drift, max_drift, dt_new);
	curandGenerateNormalDouble(gen,randomstore,2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE,0.,sqrt(dt_new));
	cudaDeviceSynchronize();
	multiply_by_timestep<<<dimgrid,dimblock>>>(drift->get_pointer(),drift_conjug->get_pointer(),dt_new);
	cudaDeviceSynchronize();
	add_noise<<<dimgrid,dimblock>>>(drift->get_pointer(),drift_conjug->get_pointer(),randomstore);	
	cudaDeviceSynchronize();
	*fields+=*drift;
	cudaDeviceSynchronize();
	*fields_conjug+=*drift_conjug;
	cudaDeviceSynchronize();
}

void Propagator::propagate_without_noise(double factor, double P, double MU)
{
	*kin=*fields;
	*kin_conjug=*fields_conjug;

	kin->fft();
	kin_conjug->fft();
	
	cudaDeviceSynchronize();
	compute_kinetic_part<<<dimgrid,dimblock>>>(kin->get_pointer(),kin_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer());
	cudaDeviceSynchronize();
	
	kin->fft_inv();
	kin_conjug->fft_inv();
	
	kin->normalize(1./double(a_x*XSIZE*YSIZE*ZSIZE));
	kin_conjug->normalize(1./double(a_x*XSIZE*YSIZE*ZSIZE));
	
	cudaDeviceSynchronize();
	calculate_drift_without_noise<<<dimgrid,dimblock>>>(fields->get_pointer(),fields_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer(),drift->get_pointer(),drift_conjug->get_pointer(), randomstore, randomstore, P, MU);
	cudaDeviceSynchronize();
	get_max_and_mean_drift(max_drift, mean_drift);
	cudaDeviceSynchronize();
	compute_new_dt(mean_drift, max_drift, dt_new);
	cudaDeviceSynchronize();
	multiply_by_timestep<<<dimgrid,dimblock>>>(drift->get_pointer(),drift_conjug->get_pointer(),dt_new);
	cudaDeviceSynchronize();
	*fields+=*drift;
	cudaDeviceSynchronize();
	*fields_conjug+=*drift_conjug;
	cudaDeviceSynchronize();
}

