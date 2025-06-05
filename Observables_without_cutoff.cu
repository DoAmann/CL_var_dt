struct which_observables{
	bool n_tot;			    //total particle number
	bool n_11;					//number of particles in Psi_11
	bool n_12;					//number of particles in Psi_12
	bool n_21;					//number of particles in Psi_21
	bool n_22;					//number of particles in Psi_22	
	bool P2;					//square of total momentum, needed for the evaluatoin of the superfluid density
	bool jrot2;					//square of the rotational part of the current density
	
	bool spectrum; 				//momentum spectrum
	bool anomalous_spectrum; 	//<a_{k}a_{-k}>
	bool dd;					//density-density correlator
	bool dispersion; 			//omega(p)
	bool spectrum_tot;			//non-angle-averaged momentum spectrum 
	bool density_tot;			//complete density in real space
	bool density_11;			//density of Psi_11-component in real space
	bool density_12;			//density of Psi_12-component in real space
	bool density_21;			//density of Psi_21-component in real space
	bool density_22;			//density of Psi_22-component in real space
	bool F_x;					//F_x = tr{Psi^dagger \sigma_x Psi}
	bool F_y;					//F_y = tr{Psi^dagger \sigma_y Psi}
	bool F_z;					//F_z = tr{Psi^dagger \sigma_z Psi}
	bool timeStepSize;
	bool drift;					//histogram of drift magnitude, needed for checking correctness of CL
	double min_drift;
	double max_drift;
	int drift_bins;			
	
	double max_momentum; 		//maximum momentum for analysis
	int bins;					//momentum bins
};



__global__ void evaluate_n_tot(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum=0.;
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]))*a_x;
	}
	
	average[j*YSIZE*ZSIZE*TAUSIZE+k*ZSIZE*TAUSIZE+l*TAUSIZE+m]=occnum;
}

__global__ void evaluate_n_11(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
		
	double occnum_11=0.;
	occnum_11 = cuCreal(cuCmul(psi_conjugate[ind(0,j,k,l,m+1)],psi[ind(0,j,k,l,m)]))*a_x;
		
	average[j*YSIZE*ZSIZE*TAUSIZE+k*ZSIZE*TAUSIZE+l*TAUSIZE+m]=occnum_11;
}

__global__ void evaluate_n_12(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
		
	double occnum_12=0.;
	occnum_12 = cuCreal(cuCmul(psi_conjugate[ind(1,j,k,l,m+1)],psi[ind(1,j,k,l,m)]))*a_x;
		
	average[j*YSIZE*ZSIZE*TAUSIZE+k*ZSIZE*TAUSIZE+l*TAUSIZE+m]=occnum_12;
}

__global__ void evaluate_n_21(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
		
	double occnum_21=0.;
	occnum_21 = cuCreal(cuCmul(psi_conjugate[ind(2,j,k,l,m+1)],psi[ind(2,j,k,l,m)]))*a_x;
		
	average[j*YSIZE*ZSIZE*TAUSIZE+k*ZSIZE*TAUSIZE+l*TAUSIZE+m]=occnum_21;
}

__global__ void evaluate_n_22(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
		
	double occnum_22=0.;
	occnum_22 = cuCreal(cuCmul(psi_conjugate[ind(3,j,k,l,m+1)],psi[ind(3,j,k,l,m)]))*a_x;
		
	average[j*YSIZE*ZSIZE*TAUSIZE+k*ZSIZE*TAUSIZE+l*TAUSIZE+m]=occnum_22;
}

__global__ void evaluate_P2(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi,double  *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum_re=0.,occnum_im=0.;
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum_re+=cuCreal(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]));
		occnum_im+=cuCimag(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]));
	}
	
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
	
	
	average[6*m*     XSIZE*YSIZE*ZSIZE + j*YSIZE*ZSIZE+k*ZSIZE+l]=occnum_re*px;
	average[(6*m+1)* XSIZE*YSIZE*ZSIZE + j*YSIZE*ZSIZE+k*ZSIZE+l]=occnum_im*px;
	average[(6*m+2)* XSIZE*YSIZE*ZSIZE + j*YSIZE*ZSIZE+k*ZSIZE+l]=occnum_re*py;
	average[(6*m+3)* XSIZE*YSIZE*ZSIZE + j*YSIZE*ZSIZE+k*ZSIZE+l]=occnum_im*py;
	average[(6*m+4)* XSIZE*YSIZE*ZSIZE + j*YSIZE*ZSIZE+k*ZSIZE+l]=occnum_re*pz;
	average[(6*m+5)* XSIZE*YSIZE*ZSIZE + j*YSIZE*ZSIZE+k*ZSIZE+l]=occnum_im*pz;
}

__global__ void evaluate_P2_q(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double  *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum=0.;
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]));
	}

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

	double p2=px*px+py*py+pz*pz;
	
	average[j*YSIZE*ZSIZE*TAUSIZE+k*ZSIZE*TAUSIZE+l*TAUSIZE+m]=occnum*p2;
}



__global__ void multiply_by_p(cuDoubleComplex *psi_conjugate, cuDoubleComplex *jk_x,cuDoubleComplex *jk_y, cuDoubleComplex *jk_z)
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
	
	jk_x[ind(0,j,k,l,m)]=make_cuDoubleComplex(psi_conjugate[ind(0,j,k,l,m+1)].x*px,psi_conjugate[ind(0,j,k,l,m+1)].y*px);
	jk_y[ind(0,j,k,l,m)]=make_cuDoubleComplex(psi_conjugate[ind(0,j,k,l,m+1)].x*py,psi_conjugate[ind(0,j,k,l,m+1)].y*py);
	jk_z[ind(0,j,k,l,m)]=make_cuDoubleComplex(psi_conjugate[ind(0,j,k,l,m+1)].x*pz,psi_conjugate[ind(0,j,k,l,m+1)].y*pz);
}

__global__ void cross_multiply_by_p(cuDoubleComplex *jk_x,cuDoubleComplex *jk_y, cuDoubleComplex *jk_z)
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
	
	double p2=px*px+py*py+pz*pz;
	
	
	cuDoubleComplex temp1,temp2,temp3;
	
	cuDoubleComplex scalprod;
	
	scalprod=make_cuDoubleComplex(jk_x[ind(0,j,k,l,m)].x*px+jk_y[ind(0,j,k,l,m)].x*py+jk_z[ind(0,j,k,l,m)].x*pz,jk_x[ind(0,j,k,l,m)].y*px+jk_y[ind(0,j,k,l,m)].y*py+jk_z[ind(0,j,k,l,m)].y*pz);
	scalprod=make_cuDoubleComplex(scalprod.x/p2,scalprod.y/p2);
	
	temp1=cuCsub(make_cuDoubleComplex(scalprod.x*px, scalprod.y*px),jk_x[ind(0,j,k,l,m)]);
	temp2=cuCsub(make_cuDoubleComplex(scalprod.x*py, scalprod.y*py),jk_y[ind(0,j,k,l,m)]);
	temp3=cuCsub(make_cuDoubleComplex(scalprod.x*pz, scalprod.y*pz),jk_z[ind(0,j,k,l,m)]);
	

	if(j==0&&k==0&&l==0)
	{
		jk_x[ind(0,j,k,l,m)]=make_cuDoubleComplex(0.,0.);
		jk_y[ind(0,j,k,l,m)]=make_cuDoubleComplex(0.,0.);
		jk_z[ind(0,j,k,l,m)]=make_cuDoubleComplex(0.,0.);
	}
	else
	{
		jk_x[ind(0,j,k,l,m)]=temp1;
		jk_y[ind(0,j,k,l,m)]=temp2;	
		jk_z[ind(0,j,k,l,m)]=temp3;	
	}
}

__global__ void evaluate_jrot2(cuDoubleComplex *jk_rot_x,cuDoubleComplex *jk_rot_y, cuDoubleComplex *jk_rot_z, double *average, int bin_number, double max_momentum)
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
	
	double momentum=sqrt(px*px+py*py+pz*pz);
	
	int index=int(momentum/max_momentum*bin_number);
	if(index<bin_number){
		atomicAdd(average+index, (cuCreal(cuCmul(jk_rot_x[ind(0,j,k,l,m)],jk_rot_x[ind(0,-j,-k,-l,m)]))
					             +cuCreal(cuCmul(jk_rot_y[ind(0,j,k,l,m)],jk_rot_y[ind(0,-j,-k,-l,m)]))
					             +cuCreal(cuCmul(jk_rot_z[ind(0,j,k,l,m)],jk_rot_z[ind(0,-j,-k,-l,m)]))) );
	}
}

__global__ void evaluate_occnum(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average, int bin_number, double max_momentum)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	if(j==0&&l==0){
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
	
	double momentum=sqrt(px*px+py*py+pz*pz);
	
	int index=int(momentum/max_momentum*bin_number);
	if(index<bin_number){
	double occnum=0.;
	
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]));
	}		
	atomicAdd(average+index,occnum);
	}
	}
}

__global__ void evaluate_angle_resolved_occnum(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum=0.;
	
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]));
	}		
	atomicAdd(average+j*YSIZE*ZSIZE+k*ZSIZE+l,occnum);
}

__global__ void evaluate_density_realspace(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum=0.;
	
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]));
	}		
	atomicAdd(average+j*YSIZE*ZSIZE+k*ZSIZE+l,occnum);
}

__global__ void evaluate_density_comp_realspace(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average, int compindex)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum=0.;
	
	occnum+=cuCreal(cuCmul(psi_conjugate[ind(compindex,j,k,l,m+1)],psi[ind(compindex,j,k,l,m)]));	

	atomicAdd(average+j*YSIZE*ZSIZE+k*ZSIZE+l,occnum);
}

__global__ void evaluate_spin_density_realspace(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average, int spin_index)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double occnum=0.;

	if(spin_index == 0){
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(2,j,k,l,m+1)],psi[ind(0,j,k,l,m)])
		+cuCmul(psi_conjugate[ind(0,j,k,l,m+1)],psi[ind(2,j,k,l,m)])
		+cuCmul(psi_conjugate[ind(3,j,k,l,m+1)],psi[ind(1,j,k,l,m)])
		+cuCmul(psi_conjugate[ind(1,j,k,l,m+1)],psi[ind(3,j,k,l,m)])
		);
	}
	else if (spin_index == 1){
		cuDoubleComplex i = make_cuDoubleComplex(0.0, 1.0);
		occnum+=cuCreal(cuCmul(i, cuCmul(psi_conjugate[ind(2,j,k,l,m+1)],psi[ind(0,j,k,l,m)])
		-cuCmul(psi_conjugate[ind(0,j,k,l,m+1)],psi[ind(2,j,k,l,m)])
		+cuCmul(psi_conjugate[ind(3,j,k,l,m+1)],psi[ind(1,j,k,l,m)])
		-cuCmul(psi_conjugate[ind(1,j,k,l,m+1)],psi[ind(3,j,k,l,m)])
		));
	}
	else if (spin_index == 2){
		occnum+=cuCreal(cuCmul(psi_conjugate[ind(0,j,k,l,m+1)],psi[ind(0,j,k,l,m)])
		+cuCmul(psi_conjugate[ind(1,j,k,l,m+1)],psi[ind(1,j,k,l,m)])
		-cuCmul(psi_conjugate[ind(2,j,k,l,m+1)],psi[ind(2,j,k,l,m)])
		-cuCmul(psi_conjugate[ind(3,j,k,l,m+1)],psi[ind(3,j,k,l,m)])
		);
	}	
	atomicAdd(average+j*YSIZE*ZSIZE+k*ZSIZE+l,occnum);
}


__global__ void evaluate_anom_occnum(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average, int bin_number, double max_momentum)
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
	
	double momentum=sqrt(px*px+py*py+pz*pz);
	
	int index=int(momentum/max_momentum*bin_number);
	if(index<bin_number){
	double occnum=0.;
	
	for(int i=0; i<COMPONENTS; i++)
	{
		occnum+=cuCreal(cuCmul(psi[ind(i,-j,-k,-l,m)],psi[ind(i,j,k,l,m)]));
	}		
	atomicAdd(average+index,occnum);
	}
}

__global__ void create_density_grid(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, cuDoubleComplex *density)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	for(int i=0; i<COMPONENTS; i++)
	{
		density[ind(i,j,k,l,m)]=cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)])*make_cuDoubleComplex(a_x,0.0);
	}
}

__global__ void evaluate_dd(cuDoubleComplex *density, double *average, int bin_number, double max_momentum)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	if(j==0&&l==0){
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
	
	double momentum=sqrt(px*px+py*py+pz*pz);
	
	int index=int(momentum/max_momentum*bin_number);
	if(index<bin_number){	
	double temp=0.;
	for(int i1=0; i1<COMPONENTS; i1++)
	{
		for(int i2=0; i2<COMPONENTS; i2++)
		{
			temp+=cuCreal(cuCmul(density[ind(i1,j,k,l,m)],density[ind(i2,-j,-k,-l,m)]));
		}	
	}	
	atomicAdd(average+index,temp);
	}
	}
}


__global__ void evaluate_disp(cuDoubleComplex *psi_conjugate, cuDoubleComplex *psi, double *average, int bin_number, double max_momentum)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	if(j==0&&l==0){
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
	
	double momentum=sqrt(px*px+py*py+pz*pz);
	
	int index=int(momentum/max_momentum*bin_number);
	if(index<bin_number){
	double disp=0.;
	
	for(int i=0; i<COMPONENTS; i++)
	{
		disp+=cuCreal(cuCsub(
		cuCadd(cuCmul(psi_conjugate[ind(i,j,k,l,m+2)],psi[ind(i,j,k,l,m)]),cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m-1)])),
		cuCadd(cuCmul(psi_conjugate[ind(i,j,k,l,m+1)],psi[ind(i,j,k,l,m)]),cuCmul(psi_conjugate[ind(i,j,k,l,m+2)],psi[ind(i,j,k,l,m-1)]))
		));
	}		
	atomicAdd(average+index,disp);	
	}
	}
}

__global__ void create_abs_grid(cuDoubleComplex *drift_conjugate, cuDoubleComplex *drift, double *result)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	for(int i=0; i<COMPONENTS; i++)
	{
		result[ind(i,j,k,l,m)]=0.5*cuCabs(drift[ind(i,j,k,l,m)]+drift_conjugate[ind(i,j,k,l,m)]);
		result[ind(i,j,k,l,m)+COMPONENTS*TAUSIZE*XSIZE*YSIZE*ZSIZE]=0.5*cuCabs(drift[ind(i,j,k,l,m)]-drift_conjugate[ind(i,j,k,l,m)]);
	}
}

class Observables
{
	public:
		Observables(ComplexLattice *fields, ComplexLattice *fields_conjug, struct which_observables which_to_evaluate);
		~Observables();
		
		struct which_observables get_which();
		
		void evaluate(double P, double MU);
		
		void write_n_tot(double *write_location);
		void write_n_11(double *write_location);
		void write_n_12(double *write_location);
		void write_n_21(double *write_location);
		void write_n_22(double *write_location);
		void write_P2(double *write_location);	
		void write_jrot2(double *write_location);					
		void write_spectrum(double *write_location);
		void write_spectrum_tot(double *write_location);
		void write_density_tot(double *write_location);
		void write_density_11(double *write_location);
		void write_density_12(double *write_location);
		void write_density_21(double *write_location);
		void write_density_22(double *write_location);
		void write_F_x(double *write_location);
		void write_F_y(double *write_location);
		void write_F_z(double *write_location);
		void write_anomalous_spectrum(double *write_location);
		void write_dd(double *write_location);
		void write_dispersion(double *write_location);
		void write_drift(int *write_location);
		
		int get_counter();
	private:
		ComplexLattice *raw_fields, *raw_fields_conjug;
		struct which_observables which;
		dim3 dimblock;
		dim3 dimgrid;
		
		int counter=0;
		
		ComplexLattice *psi_conjugate;
		ComplexLattice *psi;
		
		ComplexLattice *drift;
		ComplexLattice *drift_conjugate;
		ComplexLattice *kin;
		ComplexLattice *kin_conjugate;
		
		
		ComplexLattice *density;
		
		//improved total particle number
		double *n_tot_dev;

		double n_tot=0.;

		//improved particle number for four components
		double *n_11_dev;
		double n_11=0.;
		double *n_12_dev;
		double n_12=0.;
		double *n_21_dev;
		double n_21=0.;
		double *n_22_dev;
		double n_22=0.;
		
		//total momentum squared
		double *P2_dev, *P2_q_dev;
		
		double P2=0.;
		
		//square of the rotational part of the current density
		double *jrot2_dev;
		
		double *jrot2;
		
		//spectrum
		double *spectrum_dev;

		double *spectrum;
		
		//non-angular-averaged spectrum
		double *spectrum_tot;
		
		double *spectrum_tot_dev;
		
		//real-space density
		double *density_tot;
		
		double *density_tot_dev;

		//real-space density for four components
		double *density_11;
		double *density_11_dev;
		double *density_12;
		double *density_12_dev;
		double *density_21;
		double *density_21_dev;
		double *density_22;
		double *density_22_dev;

		//Spin density
		double *F_x;
		double *F_x_dev;
		double *F_y;
		double *F_y_dev;
		double *F_z;
		double *F_z_dev;
		
		//anomalous spectrum <a_{k}a_{-k}>
		double *anomalous_spectrum_dev;

		double *anomalous_spectrum;
		
		//density-density
		double *dd_dev;

		double *dd;
		
		//dispersion
		double *dispersion_dev;

		double *dispersion;
		
		//drift
		int *drift_counts;
		
		
		int *momentum_counter;
		
};

Observables::Observables(ComplexLattice *fields, ComplexLattice *fields_conjug, struct which_observables which_to_evaluate)
{
	raw_fields=fields;
	raw_fields_conjug=fields_conjug;

	which=which_to_evaluate;
	
	if(which.dispersion==true&&which.spectrum==false)
	{
		which.spectrum=true;
	}
	
	dim3 temp_dimblock(BLOCKSIZEX ,BLOCKSIZEY,BLOCKSIZEZ*BLOCKSIZETAU);
	dim3 temp_dimgrid(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE*TAUSIZE/(BLOCKSIZEZ*BLOCKSIZETAU)));
	dimblock=temp_dimblock;
	dimgrid=temp_dimgrid;

	
	//jrot2
	if(which.jrot2){
		cudaMalloc(&jrot2_dev,which.bins*sizeof(double));
		
		jrot2=new double [which.bins];
		
		for(int i=0; i<which.bins; i++)
		{
			jrot2[i]=0.;
		}
	}
	
	
	//spectrum
	if(which.spectrum)
	{
		cudaMalloc(&spectrum_dev,which.bins*sizeof(double));

		spectrum=new double [which.bins];

		for(int i=0; i<which.bins; i++)
		{
			spectrum[i]=0.;
		}
	}
	
	//non-angular-averaged spectrum 
	if(which.spectrum_tot)
	{
		cudaMalloc(&spectrum_tot_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		spectrum_tot=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			spectrum_tot[i]=0.;
		}
	}
	
	//real-space density 
	if(which.density_tot)
	{
		cudaMalloc(&density_tot_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		density_tot=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_tot[i]=0.;
		}
	}
	//real-space density for four components
	if(which.density_11)
	{
		cudaMalloc(&density_11_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		density_11=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_11[i]=0.;
		}
	}
	if(which.density_12)
	{
		cudaMalloc(&density_12_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		density_12=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_12[i]=0.;
		}
	}
	if(which.density_21)
	{
		cudaMalloc(&density_21_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		density_21=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_21[i]=0.;
		}
	}
	if(which.density_22)
	{
		cudaMalloc(&density_22_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		density_22=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_22[i]=0.;
		}
	}
	if(which.F_x)
	{
		cudaMalloc(&F_x_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		F_x=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			F_x[i]=0.;
		}
	}
	if(which.F_y)
	{
		cudaMalloc(&F_y_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		F_y=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			F_y[i]=0.;
		}
	}
	if(which.F_z)
	{
		cudaMalloc(&F_z_dev,XSIZE*YSIZE*ZSIZE*sizeof(double));

		F_z=new double [XSIZE*YSIZE*ZSIZE];

		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			F_z[i]=0.;
		}
	}
	
	//anomalous spectrum
	if(which.anomalous_spectrum)
	{
		cudaMalloc(&anomalous_spectrum_dev,which.bins*sizeof(double));

		anomalous_spectrum=new double [which.bins];

		for(int i=0; i<which.bins; i++)
		{
			anomalous_spectrum[i]=0.;
		}
	}
	
	//density-density
	if(which.dd)
	{
		cudaMalloc(&dd_dev,which.bins*sizeof(double));

		dd=new double [which.bins];

		for(int i=0; i<which.bins; i++)
		{
			dd[i]=0.;
		}
	}
	
	//dispersion
	if(which.dispersion)
	{
		cudaMalloc(&dispersion_dev,which.bins*sizeof(double));

		dispersion=new double [which.bins];

		for(int i=0; i<which.bins; i++)
		{
			dispersion[i]=0.;
		}
	}
	
	
	if(which.drift)
	{
		drift_counts=new int [which.drift_bins];
		for(int i=0; i<which.drift_bins; i++)
		{
			drift_counts[i]=0;
		}
	}
	

	psi_conjugate=new ComplexLattice(true);
	psi=new ComplexLattice(false);
	
	momentum_counter=new int [which.bins];
	for(int i=0; i<which.bins; i++)
	{
		momentum_counter[i]=0;
	}
	double momentum=0.; int index;
	double px,py,pz;
	for(int j=0; j<XSIZE; j++){for(int k=0; k<YSIZE; k++){for(int l=0; l<ZSIZE; l++){

	if(j==0&&l==0){
	
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
	
	momentum=sqrt(px*px+py*py+pz*pz);
		
		index=int(momentum/which.max_momentum*which.bins);
		if(index<which.bins){
			momentum_counter[index]++;
		}
	}
	}}}

	
	if(which.dd)
	{
		density=new ComplexLattice(true);
	}
	
	if(which.drift)
	{
		drift_conjugate=new ComplexLattice(true);
	    drift=new ComplexLattice(false);
		kin_conjugate=new ComplexLattice(true);
	    kin=new ComplexLattice(false);

	}
}

Observables::~Observables()
{
	if(which.jrot2)
	{
		cudaFree(jrot2_dev);
	}

	delete psi;
	delete psi_conjugate;
	
	if(which.drift)
	{
		delete drift;
		delete drift_conjugate;
		delete kin;
		delete kin_conjugate;
	}

	if(which.spectrum)
	{
		cudaFree(spectrum_dev);
	}
	if(which.spectrum_tot)
	{
		cudaFree(spectrum_tot_dev);
	}
	if(which.density_tot)
	{
		cudaFree(density_tot_dev);
	}
	if(which.density_11)
	{
		cudaFree(density_11_dev);
	}
	if(which.density_12)
	{
		cudaFree(density_12_dev);
	}
	if(which.density_21)
	{
		cudaFree(density_21_dev);
	}
	if(which.density_22)
	{
		cudaFree(density_22_dev);
	}
	if(which.F_x)
	{
		cudaFree(F_x_dev);
	}
	if(which.F_y)
	{
		cudaFree(F_y_dev);
	}
	if(which.F_z)
	{
		cudaFree(F_z_dev);
	}
	if(which.anomalous_spectrum)
	{
		cudaFree(anomalous_spectrum_dev);
	}
	if(which.dd)
	{
		cudaFree(dd_dev);
		delete density;
	}
	if(which.dispersion)
	{
		cudaFree(dispersion_dev);
	}
	if(which.drift)
	{
		delete drift_counts;
	}
}

struct which_observables Observables::get_which()
{ 
	return which;
}
	
void Observables::evaluate(double P, double MU)
{
	*psi_conjugate=*raw_fields_conjug;
	*psi=*raw_fields;
	
	
	if(which.dd)
	{
		cudaDeviceSynchronize();
		create_density_grid<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), density->get_pointer());
		cudaDeviceSynchronize();
	}
	ComplexLattice *psi_rs;
	if(which.jrot2)
	{
		 psi_rs=new ComplexLattice(false);
		*psi_rs = *psi;
	}
	
	if(which.drift)
	{
		*kin=*psi;
		*kin_conjugate=*psi_conjugate;

		kin->fft();
		kin_conjugate->fft();
		
		cudaDeviceSynchronize();
		compute_kinetic_part<<<dimgrid,dimblock>>>(kin->get_pointer(),kin_conjugate->get_pointer(),kin->get_pointer(),kin_conjugate->get_pointer());
		cudaDeviceSynchronize();
		
		kin->fft_inv();
		kin_conjugate->fft_inv();
		
		kin->normalize(1./double(a_x*XSIZE*YSIZE*ZSIZE));
		kin_conjugate->normalize(1./double(a_x*XSIZE*YSIZE*ZSIZE));
		
		cudaDeviceSynchronize();
		calculate_drift_without_noise<<<dimgrid,dimblock>>>(psi->get_pointer(),psi_conjugate->get_pointer(),kin->get_pointer(),kin_conjugate->get_pointer(),drift->get_pointer(),drift_conjugate->get_pointer(),1./DT, P, MU);
		cudaDeviceSynchronize();
		
		double *modulus_drift;
		cudaMalloc(&modulus_drift,2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));
		
		cudaDeviceSynchronize();
		create_abs_grid<<<dimgrid,dimblock>>>(drift_conjugate->get_pointer(), drift->get_pointer(), modulus_drift);
		cudaDeviceSynchronize();
		
		
		thrust::device_ptr<double> thrust_ptr_modulus_drift(modulus_drift);
        thrust::device_vector<double> thrust_vec_modulus_drift(thrust_ptr_modulus_drift, thrust_ptr_modulus_drift +2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		thrust::device_vector<double>::iterator iter=thrust::max_element(thrust_vec_modulus_drift.begin(), thrust_vec_modulus_drift.end());
		
		double value=*iter;
		
		int b=int((value-which.min_drift)/(which.max_drift-which.min_drift)*which.drift_bins);
		if(b>=0&&b<which.drift_bins)
		{
			drift_counts[b]+=1;
		}
		
		cudaFree(modulus_drift);
	}
	
	if(which.density_tot)
	{
		set_multiple_device_variables(density_tot_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_density_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), density_tot_dev);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_tot[i]+=get_device_variable(density_tot_dev+i);
		}
	}
	if(which.density_11)
	{
		set_multiple_device_variables(density_11_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_density_comp_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), density_11_dev, 0);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_11[i]+=get_device_variable(density_11_dev+i);
		}
	}
	if(which.density_12)
	{
		set_multiple_device_variables(density_12_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_density_comp_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), density_12_dev, 1);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_12[i]+=get_device_variable(density_12_dev+i);
		}
	}
	if(which.density_21)
	{
		set_multiple_device_variables(density_21_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_density_comp_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), density_21_dev, 2);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_21[i]+=get_device_variable(density_21_dev+i);
		}
	}
	if(which.density_22)
	{
		set_multiple_device_variables(density_22_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_density_comp_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), density_22_dev, 3);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			density_22[i]+=get_device_variable(density_22_dev+i);
		}
	}
	if(which.F_x)
	{
		set_multiple_device_variables(F_x_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_spin_density_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), F_x_dev, 0);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			F_x[i]+=get_device_variable(F_x_dev+i);
		}
	}
	if(which.F_y)
	{
		set_multiple_device_variables(F_y_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_spin_density_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), F_y_dev, 1);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			F_y[i]+=get_device_variable(F_y_dev+i);
		}
	}
	if(which.F_z)
	{
		set_multiple_device_variables(F_z_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_spin_density_realspace<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), F_z_dev, 2);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			F_z[i]+=get_device_variable(F_z_dev+i);
		}
	}

	

	psi_conjugate->fft();
	psi->fft();
	
	
	
	
	
	if(which.dd)
	{
		density->fft();
	}

	if(which.n_tot)
	{
		cudaMalloc(&n_tot_dev,sizeof(double)*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		cudaDeviceSynchronize();
		evaluate_n_tot<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), n_tot_dev);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> thrust_ptr_n_tot(n_tot_dev);
        thrust::device_vector<double> thrust_vec_n_tot(thrust_ptr_n_tot, thrust_ptr_n_tot +XSIZE*YSIZE*ZSIZE*TAUSIZE);
		n_tot+=thrust::reduce(thrust_vec_n_tot.begin(), thrust_vec_n_tot.end());
		cudaFree(n_tot_dev);

	}

	if(which.n_11)
	{
		cudaMalloc(&n_11_dev,sizeof(double)*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		cudaDeviceSynchronize();
		evaluate_n_11<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), n_11_dev);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> thrust_ptr_n_11(n_11_dev);
		thrust::device_vector<double> thrust_vec_n_11(thrust_ptr_n_11, thrust_ptr_n_11 +XSIZE*YSIZE*ZSIZE*TAUSIZE);
		n_11+=thrust::reduce(thrust_vec_n_11.begin(), thrust_vec_n_11.end());
		cudaFree(n_11_dev);
	
	}
	if(which.n_12)
	{
		cudaMalloc(&n_12_dev,sizeof(double)*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		cudaDeviceSynchronize();
		evaluate_n_12<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), n_12_dev);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> thrust_ptr_n_12(n_12_dev);
		thrust::device_vector<double> thrust_vec_n_12(thrust_ptr_n_12, thrust_ptr_n_12 +XSIZE*YSIZE*ZSIZE*TAUSIZE);
		n_12+=thrust::reduce(thrust_vec_n_12.begin(), thrust_vec_n_12.end());
		cudaFree(n_12_dev);
	
	}
	if(which.n_21)
	{
		cudaMalloc(&n_21_dev,sizeof(double)*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		cudaDeviceSynchronize();
		evaluate_n_21<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), n_21_dev);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> thrust_ptr_n_21(n_21_dev);
		thrust::device_vector<double> thrust_vec_n_21(thrust_ptr_n_21, thrust_ptr_n_21 +XSIZE*YSIZE*ZSIZE*TAUSIZE);
		n_21+=thrust::reduce(thrust_vec_n_21.begin(), thrust_vec_n_21.end());
		cudaFree(n_21_dev);
	
	}
	if(which.n_22)
	{
		cudaMalloc(&n_22_dev,sizeof(double)*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		cudaDeviceSynchronize();
		evaluate_n_22<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), n_22_dev);
		cudaDeviceSynchronize();
		thrust::device_ptr<double> thrust_ptr_n_22(n_22_dev);
		thrust::device_vector<double> thrust_vec_n_22(thrust_ptr_n_22, thrust_ptr_n_22 +XSIZE*YSIZE*ZSIZE*TAUSIZE);
		n_22+=thrust::reduce(thrust_vec_n_22.begin(), thrust_vec_n_22.end());
		cudaFree(n_22_dev);
	
	}
	
	if(which.P2)
	{
		cudaMalloc(&P2_dev,6*XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));
		cudaMalloc(&P2_q_dev,XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));

		cudaDeviceSynchronize();
		evaluate_P2<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), P2_dev);
		cudaDeviceSynchronize();

		thrust::device_ptr<double> thrust_ptr_P2(P2_dev);
        thrust::device_vector<double> thrust_vec_P2(thrust_ptr_P2, thrust_ptr_P2 +6*XSIZE*YSIZE*ZSIZE*TAUSIZE);
		thrust::device_vector<double> P2_temp1(6*TAUSIZE);
		thrust::reduce_by_key(thrust::device, thrust::make_transform_iterator(thrust::counting_iterator<int>(0), thrust::placeholders::_1/(a_x*XSIZE*YSIZE*ZSIZE)), thrust::make_transform_iterator(thrust::counting_iterator<int>(6*TAUSIZE*XSIZE*YSIZE*ZSIZE), thrust::placeholders::_1/(a_x*XSIZE*YSIZE*ZSIZE)), thrust_vec_P2.begin(),thrust::discard_iterator<int>(), P2_temp1.begin());
	

		cudaDeviceSynchronize();        
	    evaluate_P2_q<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), P2_q_dev);
		cudaDeviceSynchronize();

		thrust::device_ptr<double> thrust_ptr_P2_q(P2_q_dev);
        thrust::device_vector<double> thrust_vec_P2_q(thrust_ptr_P2_q, thrust_ptr_P2_q +XSIZE*YSIZE*ZSIZE*TAUSIZE);	
		double P2_temp2;
		P2_temp2=thrust::reduce(thrust_vec_P2_q.begin(), thrust_vec_P2_q.end());
		
		for(int i=0; i<3*TAUSIZE; i++)
		{
			P2+=(mysquare_h(P2_temp1[2*i])-mysquare_h(P2_temp1[2*i+1]))/mysquare_h(a_x*XSIZE*YSIZE*ZSIZE);
		}
		P2+=P2_temp2/(a_x*XSIZE*YSIZE*ZSIZE);
		
		cudaFree(P2_dev);
		cudaFree(P2_q_dev);
	}
	
	
	if(which.jrot2)
	{
		ComplexLattice *jk_x=new ComplexLattice(true);
		ComplexLattice *jk_y=new ComplexLattice(true);
		ComplexLattice *jk_z=new ComplexLattice(true);
		
		cudaDeviceSynchronize();
		multiply_by_p<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(),jk_x->get_pointer(),jk_y->get_pointer(),jk_z->get_pointer());
		cudaDeviceSynchronize();
		
		jk_x->fft_inv();
		jk_y->fft_inv();	
		jk_z->fft_inv();
		
		*jk_x *= *psi_rs;
		*jk_y *= *psi_rs;
		*jk_z *= *psi_rs;
		
		jk_x->fft();
		jk_y->fft();	
		jk_z->fft();	
		
		cudaDeviceSynchronize();
		cross_multiply_by_p<<<dimgrid,dimblock>>>(jk_x->get_pointer(),jk_y->get_pointer(),jk_z->get_pointer());
		cudaDeviceSynchronize();	
		
		set_multiple_device_variables(jrot2_dev,0.,which.bins);
		cudaDeviceSynchronize();
		evaluate_jrot2<<<dimgrid,dimblock>>>(jk_x->get_pointer(),jk_y->get_pointer(),jk_z->get_pointer(),jrot2_dev, which.bins, which.max_momentum);
		cudaDeviceSynchronize();
		
		for(int i=0; i<which.bins; i++)
		{
			jrot2[i]+=get_device_variable(jrot2_dev+i);
		}
		
		delete jk_x; delete jk_y; delete jk_z; delete psi_rs; 
	}
		
	if(which.spectrum)
	{
		set_multiple_device_variables(spectrum_dev,0.,which.bins);
		cudaDeviceSynchronize();
		evaluate_occnum<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), spectrum_dev, which.bins, which.max_momentum);
		cudaDeviceSynchronize();
		for(int i=0; i<which.bins; i++)
		{
			spectrum[i]+=get_device_variable(spectrum_dev+i);
		}
	}
	
	if(which.spectrum_tot)
	{
		set_multiple_device_variables(spectrum_tot_dev,0.,XSIZE*YSIZE*ZSIZE);
		cudaDeviceSynchronize();
		evaluate_angle_resolved_occnum<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), spectrum_tot_dev);
		cudaDeviceSynchronize();
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			spectrum_tot[i]+=get_device_variable(spectrum_tot_dev+i);
		}
	}
	
	if(which.anomalous_spectrum)
	{
		set_multiple_device_variables(anomalous_spectrum_dev,0.,which.bins);
		cudaDeviceSynchronize();
		evaluate_anom_occnum<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), anomalous_spectrum_dev, which.bins, which.max_momentum);
		cudaDeviceSynchronize();
		for(int i=0; i<which.bins; i++)
		{
			anomalous_spectrum[i]+=get_device_variable(anomalous_spectrum_dev+i);
		}
	}

	if(which.dd)
	{
		set_multiple_device_variables(dd_dev,0.,which.bins);
		cudaDeviceSynchronize();
		evaluate_dd<<<dimgrid,dimblock>>>(density->get_pointer(), dd_dev, which.bins, which.max_momentum);
		cudaDeviceSynchronize();
		for(int i=0; i<which.bins; i++)
		{
			dd[i]+=get_device_variable(dd_dev+i);
		}
	}
	
	if(which.dispersion)
	{
		set_multiple_device_variables(dispersion_dev,0.,which.bins);
		cudaDeviceSynchronize();
		evaluate_disp<<<dimgrid,dimblock>>>(psi_conjugate->get_pointer(), psi->get_pointer(), dispersion_dev, which.bins, which.max_momentum);
		cudaDeviceSynchronize();
		for(int i=0; i<which.bins; i++)
		{
			dispersion[i]+=get_device_variable(dispersion_dev+i);
		}
	}
	
	if(which.drift)
	{

	}
	
	counter++;
}


void Observables::write_n_tot(double *write_location)
{
	*write_location=n_tot/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
}

void Observables::write_n_11(double *write_location)
{
	*write_location=n_11/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
}
void Observables::write_n_12(double *write_location)
{
	*write_location=n_12/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
}
void Observables::write_n_21(double *write_location)
{
	*write_location=n_21/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
}
void Observables::write_n_22(double *write_location)
{
	*write_location=n_22/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
}

void Observables::write_P2(double *write_location)
{
	*write_location=P2/double(TAUSIZE)/double(counter);
}


void Observables::write_jrot2(double *write_location)
{
	for(int i=0; i<which.bins; i++)
	{
		write_location[i]=jrot2[i]/double(momentum_counter[i])/double(TAUSIZE)/double(counter)/mycube_h(XSIZE*YSIZE*ZSIZE)*2.;
	}
}

void Observables::write_spectrum(double *write_location)
{
	for(int i=0; i<which.bins; i++)
	{
		write_location[i]=spectrum[i]/double(momentum_counter[i])/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
	}
}

void Observables::write_spectrum_tot(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=spectrum_tot[i]/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
	}
}

void Observables::write_density_tot(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=density_tot[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_density_11(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=density_11[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_density_12(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=density_12[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_density_21(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=density_21[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_density_22(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=density_22[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_F_x(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=F_x[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_F_y(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=F_y[i]/double(TAUSIZE)/double(counter);
	}
}
void Observables::write_F_z(double *write_location)
{
	for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
	{
		write_location[i]=F_z[i]/double(TAUSIZE)/double(counter);
	}
}

void Observables::write_anomalous_spectrum(double *write_location)
{
	for(int i=0; i<which.bins; i++)
	{
		write_location[i]=anomalous_spectrum[i]/double(momentum_counter[i])/double(TAUSIZE)/double(counter)/(XSIZE*YSIZE*ZSIZE);
	}
}

void Observables::write_dd(double *write_location)
{
	for(int i=0; i<which.bins; i++)
	{
		write_location[i]=dd[i]/double(momentum_counter[i])/double(TAUSIZE)/double(counter)/mysquare_h(XSIZE*YSIZE*ZSIZE);
	}
}

void Observables::write_dispersion(double *write_location)
{
	for(int i=0; i<which.bins; i++)
	{
        if(-dispersion[i]/spectrum[i]>0)
        {
			write_location[i]=sqrt(-dispersion[i]/spectrum[i])/EPS;
		}
	    else
		{
			write_location[i]=0.;
		}
	}

}

void Observables::write_drift(int *write_location)
{
	for(int i=0; i<which.drift_bins; i++)
	{
		write_location[i]=drift_counts[i];
	}
}

int Observables::get_counter()
{
	return counter;
}
