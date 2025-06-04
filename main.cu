#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <cuComplex.h>
#include <cufft.h>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <complex>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <sstream>
#include <fstream>
#include <errno.h>
#include <filesystem>
#include <map>
using namespace std;
#include "Parameters.h"

#include "Parameters.h"
#include "Helpers.cu"
#include "Lattice.cu"
#include "Propagator_without_cutoff.cu"
#include "Observables_without_cutoff.cu"
#include "SaveResults.cu"

#define mf false

std::map<std::string, double> read_params(const std::string& filename) {
    std::map<std::string, double> params;
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string key;
        double value;
        if (std::getline(iss, key, '=') && iss >> value) {
            params[key] = value;
        }
    }
    return params;
}

int main(int argc, char** argv)
{
	int devnum = 0;                            
    if(argc > 1)
       {devnum = atoi(argv[1]);}
    cudaSetDevice(devnum);
	double timer=time(NULL);

	struct which_observables which;
	which.n_tot=true;
	which.n_11=true; which.n_12=true; which.n_21=true; which.n_22=true;
	which.P2=false;
	which.jrot2=false;
	which.spectrum=false;
	which.dispersion=false;
	which.dd=false;
	which.anomalous_spectrum=false;
	which.density_tot=false; which.density_11=false; which.density_12=false; which.density_21=false; which.density_22=false;
	which.F_x=true; which.F_y = true; which.F_z=true;
	which.spectrum_tot=false;
	which.drift=false;; which.dt_new=true;
	which.max_momentum= M_PI * sqrt(2.0);
	which.bins=284;
	which.min_drift=0.;
	which.max_drift=100;

	auto params = read_params("params.txt"); 
	double MU = 0.0;
	double P  = 0.0;
	
	try {
		MU = params.at("mu");
		P  = params.at("P");
	} catch (const std::out_of_range& e) {
		std::cerr << "Fehler: params.txt unvollständig (mu/P fehlen?)" << std::endl;
		return 1;
	}

unsigned long long seedvalue = static_cast<unsigned long long>(params["seed"]);
unsigned long long seedvalue2 = static_cast<unsigned long long>(params["seed2"]);
	
	ComplexLattice *fields=new ComplexLattice(false);
	ComplexLattice *fields_conjug=new ComplexLattice(true);
	
	ComplexLattice *drift=new ComplexLattice(false);
	ComplexLattice *drift_conjug=new ComplexLattice(true);

	//ComplexLatticeHost *fields_host=new ComplexLatticeHost();
	//ComplexLatticeHost *fields_conjug_host=new ComplexLatticeHost();

	Propagator *prop=new Propagator(fields,fields_conjug,drift,drift_conjug,seedvalue, seedvalue2);
	Observables *obs=new Observables(fields,fields_conjug,which);
	
	SaveResults * save= new SaveResults(obs,seedvalue);
	if(!save->make_directory(P, MU)){
		return 1;
	}
	save->save_parameters(P, MU);
	save->make_directory_rawdata();

	

	if(mf)
	{
	    fields->set_mean_field(sqrt(abs(MU)/G/COMPONENTS),0.);
	    fields_conjug->set_mean_field(sqrt(abs(MU)/G/COMPONENTS),0.);
	}
	else
	{
	    fields->set_mean_field(0.,0.);
	    fields_conjug->set_mean_field(0.,0.);
	}

	
	cout<<"Started simulation with parameters"<<endl
		  <<"T "<<1./(TAUSIZE*EPS)<<endl
		  <<"mu "<<MU<<endl
		  <<"g "<<G<<endl
		  <<"M "<<M<<endl
		  <<"SIGMA "<<SIGMA<<endl
		  <<"D "<<D<<endl
		  <<"OMEGAX "<<OMEGAX<<endl
		  <<"OMEGAY "<<OMEGAY<<endl
		  <<"OMEGAZ "<<OMEGAZ<<endl
		  <<"P "<<P<<endl
		  <<"components "<<COMPONENTS<<endl
		  <<"lattice "<<XSIZE<<"x"<<YSIZE<<"x"<<ZSIZE<<"x"<<TAUSIZE<<endl
		  <<"dt "<<DT<<endl
		  <<"steps "<<STEPS<<endl
		  <<"seed "<<seedvalue<<endl;

	for(int i=0; i<STEPS; i++)
	{
		bool is_snapshot_step = ((i % SNAPSHOT_TRIGGER) == 0) && i != 0 && i != STEPS - 1;
		if (is_snapshot_step || i == 1)
			{
				save->save_snapshot(P, MU);
			}
			
			prop->propagate(P, MU);
			
			if(i%EVALUATE_TRIGGER==0){
				obs->evaluate(P, MU);
			}       
	}
	save->save_snapshot(P, MU);
	cout<<"Finished simulation after computation time of "<<time(NULL)-timer<<" seconds."<<endl;

	return 0;
}
