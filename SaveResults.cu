//checks whether a folder exists
bool folder_exists(string foldername)
{
	struct stat st;
	if(stat(foldername.c_str(),&st) == 0){
		if(S_ISDIR(st.st_mode)== 0)
		{
			return false;
		}
		else
		{
			return true;
		}
    }
    else
    {
		return false;
	}
}


class SaveResults
{
	public:
		SaveResults(Observables *observables, unsigned long long seedval);
		bool make_directory(double P, double MU);
		bool make_directory(string dirname);
		bool make_directory_rawdata();
		void save_parameters(double P, double MU);
		void save_snapshot(double P, double MU);
		void save_lattice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug);
		void save_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int tauindex);
		void save_lattice_slices(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int min_tauindex,int max_tauindex);
		void read_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, string filename);
		
	private:
		Observables *obs;
		struct which_observables which;
		string directory;
		string directory_rawdata;
		unsigned long long seedvalue;
};

SaveResults::SaveResults(Observables *observables, unsigned long long seedval)
{
	obs=observables;
	which=obs->get_which();
	seedvalue=seedval;
}


bool SaveResults::make_directory(double P, double MU)
{
	stringstream dirname;

	dirname<<"./DATA/BoseCL_Sim_mu_"<<MU<<"_P_"<<P<<"/Run";
	
	//avoid to create an existing folder
	for(int r=1; r<1000; r++)
	{
		if(folder_exists(dirname.str()+to_string(r))==false)
		{
			dirname<<r;
			break;
		}
	}
	std::filesystem::path dirPath = dirname.str();
	
	// Create the directory (and all intermediate directories, if necessary)
    try {
        if (std::filesystem::create_directories(dirPath)) {
            std::cout << "Directory created successfully: " << dirPath << std::endl;
			directory=dirname.str();
			return true;
        } 
		else {
            std::cout << "Directory already exists: " << dirPath << std::endl;
			return false;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
		return false;
    }
}
	
	/* OLD 
	if(mkdir(directory.c_str(),0755)<0)
	{
		cout<<"Directory could not be created! Error code "<<errno<<endl;
		return false;
	}
	else
	{
		cout<<"Directory with name "<<directory.c_str()<<" was successfully created"<<endl;
		return true;
	}
	*/

bool SaveResults::make_directory(string dirname)
{
	
	//avoid to create an existing folder
	for(int r=1; r<1000; r++)
	{
		if(folder_exists(dirname+to_string(r))==false)
		{
			dirname+=to_string(r);
			break;
		}
	}
	std::filesystem::path dirPath = dirname;
	
	// Create the directory (and all intermediate directories, if necessary)
    try {
        if (std::filesystem::create_directories(dirPath)) {
            std::cout << "Directory created successfully: " << dirPath << std::endl;
			directory=dirname;
			return true;
        } 
		else {
            std::cout << "Directory already exists: " << dirPath << std::endl;
			return false;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
		return false;
    }
}

	/* OLD 
	for(int r=1; r<1000; r++)
	{
		if(folder_exists(dirname+to_string(r))==false)
		{
			directory+=to_string(r);
			break;
		}
	}
	if(mkdir(directory.c_str(),0755)<0)
	{
		cout<<"Directory could not be created!"<<endl;
		return false;
	}
	else
	{
		cout<<"Directory with name "<<directory.c_str()<<" was successfully created"<<endl;
		return true;
	}
	*/

bool SaveResults::make_directory_rawdata()
{
	directory_rawdata=directory+"/rawdata";
	if(mkdir(directory_rawdata.c_str(),0755)<0)
	{
		cout<<"Directory for raw data could not be created!"<<endl;
		return false;
	}
	else
	{
		cout<<"Directory for raw data was successfully created"<<endl;
		return true;
	}
}	

void SaveResults::save_parameters(double P, double MU)
{
	fstream output;
	stringstream temp;
	temp<<directory<<"/Parameters.txt";		
	output.open(temp.str(),fstream::out);
	output<<"T\t"<<1./(TAUSIZE*EPS)<<endl
		  <<"mu\t"<<MU<<endl
		  <<"g\t"<<G<<endl
		  <<"M\t"<<M<<endl
		  <<"SIGMA\t"<<SIGMA<<endl
		  <<"D\t"<<D<<endl
		  <<"OMEGAX\t"<<OMEGAX<<endl
		  <<"OMEGAY\t"<<OMEGAY<<endl
		  <<"OMEGAZ\t"<<OMEGAZ<<endl
		  <<"P\t"<<P<<endl
		  <<"Q\t"<<Q<<endl
		  <<"components\t"<<COMPONENTS<<endl
		  <<"lattice\t"<<XSIZE<<"x"<<YSIZE<<"x"<<ZSIZE<<"x"<<TAUSIZE<<endl
		  <<"dt\t"<<DT<<endl
		  <<"steps\t"<<STEPS<<endl
		  <<"SNAPSHOT_TRIGGER\t"<<SNAPSHOT_TRIGGER<<endl
		  <<"EVALUATE_TRIGGER\t"<<EVALUATE_TRIGGER<<endl
		  <<"seed\t"<<seedvalue;
	output.close();
	cout<<"Saved parameters"<<endl;
}

void SaveResults::save_snapshot(double P, double MU)
{
	int number=obs->get_counter();
	
	fstream output;
	stringstream temp;

	/////////////
	// SCALARS //
	/////////////
	stringstream directory_scalars;
	directory_scalars<<"./DATA/BoseCL_Sim_mu_"<<MU<<"_P_"<<P; 
	temp<<directory_scalars.str()<<"/Skalars.txt";		
	output.open(temp.str(),fstream::app);
	
	double n_total, num_11, num_12, num_21, num_22, P2;
	string str_to_append;
	if(which.n_tot){
		obs->write_n_tot(&n_total);
		str_to_append = str_to_append + to_string(n_total) + "\t";
	}
	else{
		str_to_append = "\t\t";
	}
	if(which.n_11){
		obs->write_n_11(&num_11);
		str_to_append = str_to_append + to_string(num_11) + "\t";
	}
	else{
		str_to_append = str_to_append + "\t\t";
	}
	if(which.n_12){
		obs->write_n_12(&num_12);
		str_to_append = str_to_append + to_string(num_12) + "\t";
	}
	else{
		str_to_append = str_to_append + "\t\t";
	}
	if(which.n_21){
		obs->write_n_21(&num_21);
		str_to_append = str_to_append + to_string(num_21) + "\t";
	}
	else{
		str_to_append = str_to_append + "\t\t";
	}
	if(which.n_22){
		obs->write_n_22(&num_22);
		str_to_append = str_to_append + to_string(num_22) + "\t";
	}
	else{
		str_to_append = str_to_append + "\t\t";
	}

	if(which.P2){
		obs->write_P2(&P2);
		str_to_append = str_to_append + to_string(P2) + "\t";
	}

	else{
		str_to_append = str_to_append + "\t\t";
	}

	output << str_to_append << endl;


	//OLD SAVE
	/*
	fstream output_1;
	stringstream temp_1;

	temp_1<<directory<<"/Scalars_"<<number<<".txt";	
	output_1.open(temp_1.str(),fstream::out);
	
	//double n_total, P2;
	
	if(which.n_tot){
		obs->write_n_tot(&n_total);
		cout<<n_total;
		output_1<<"Total particle number: "<<n_total<<endl;
	}
	if(which.P2){
		obs->write_P2(&P2);
		cout<<P2;
  	output_1<<"Total momentum squared: "<<P2<<endl;
	}
	
	
	if(output.good() && output_1.good())
	*/
	if(output.good())
	{
		cout<<"Saved scalars at time "<<number<<endl;
	}
	else
	{
		cout<<"Saving scalars at time "<<number<<" failed"<<endl;
	}
	output.close();
	// output_1.close();

	//////////////
	// SPECTRUM //
	//////////////

	if(which.spectrum)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum.txt";		
		output.open(temp.str(),fstream::app);
		double *spectrum=new double[which.bins];
		obs->write_spectrum(spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<spectrum[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		
		
		/* OLD
		temp<<directory<<"/Spectrum_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *spectrum=new double[which.bins];
		obs->write_spectrum(spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<spectrum[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}
	// SPECTRUM TOT (not angular averaging)
	if(which.spectrum_tot)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_tot.txt";		
		output.open(temp.str(),fstream::app);
		double *spectrum_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_spectrum_tot(spectrum_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<spectrum_tot[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved angular-resolved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving angular-resolved spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();



		/*OLD
		temp<<directory<<"/Spectrum_tot_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *spectrum_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_spectrum_tot(spectrum_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<spectrum_tot[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved angular-resolved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving angular-resolved spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	//  DENSITY TOT (not angular averaging)
	if(which.density_tot)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_tot.txt";		
		output.open(temp.str(),fstream::app);
		double *density_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_tot(density_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_tot[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space density at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density at time "<<number<<" failed"<<endl;
		}
		output.close();

		/*OLD
		temp<<directory<<"/Spectrum_tot_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *density_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_tot(density_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_tot[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved real-space density at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density at time "<<number<<" failed"<<endl;
		}
		output.close();
		 */
	}

	if(which.density_11)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_11.txt";		
		output.open(temp.str(),fstream::app);
		double *density_11=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_11(density_11);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_11[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space density_11 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density_11 at time "<<number<<" failed"<<endl;
		}
		output.close();
	}
	if(which.density_12)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_12.txt";		
		output.open(temp.str(),fstream::app);
		double *density_12=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_12(density_12);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_12[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space density_12 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density_12 at time "<<number<<" failed"<<endl;
		}
		output.close();
	}
	if(which.density_21)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_21.txt";		
		output.open(temp.str(),fstream::app);
		double *density_21=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_21(density_21);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_21[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space density_21 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density_21 at time "<<number<<" failed"<<endl;
		}
		output.close();
	}
	if(which.density_22)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_22.txt";		
		output.open(temp.str(),fstream::app);
		double *density_22=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_22(density_22);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_22[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space density_22 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density_22 at time "<<number<<" failed"<<endl;
		}
		output.close();
	}

	if(which.F_x)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/F_x.txt";		
		output.open(temp.str(),fstream::app);
		double *F_x=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_F_x(F_x);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<F_x[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space F_x at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space F_x at time "<<number<<" failed"<<endl;
		}
		output.close();
	}

	if(which.F_y)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/F_y.txt";		
		output.open(temp.str(),fstream::app);
		double *F_y=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_F_y(F_y);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<F_y[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space F_y at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space F_y at time "<<number<<" failed"<<endl;
		}
		output.close();
	}

	if(which.F_z)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/F_z.txt";		
		output.open(temp.str(),fstream::app);
		double *F_z=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_F_z(F_z);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<F_z[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space F_z at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space F_z at time "<<number<<" failed"<<endl;
		}
		output.close();
	}



	// ANOMALOUS SPETRUM 
	if(which.anomalous_spectrum)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Anomalous_Spectrum.txt";		
		output.open(temp.str(),fstream::app);
		double *anomalous_spectrum=new double[which.bins];
		obs->write_anomalous_spectrum(anomalous_spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<anomalous_spectrum[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved anomalous spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving anomalous spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD
		temp<<directory<<"/Anomalous_Spectrum_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *anomalous_spectrum=new double[which.bins];
		obs->write_anomalous_spectrum(anomalous_spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<anomalous_spectrum[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved anomalous spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving anomalous spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	// DISPERSION
	if(which.dispersion)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Dispersion.txt";		
		output.open(temp.str(),fstream::app);
		double *dispersion=new double[which.bins];
		obs->write_dispersion(dispersion);
		for(int i=0; i<which.bins; i++)
		{
			output<<dispersion[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved dispersion at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dispersion at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/Dispersion_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *dispersion=new double[which.bins];
		obs->write_dispersion(dispersion);
		for(int i=0; i<which.bins; i++)
		{
			output<<dispersion[i]<<endl;
		}
		
		if(output.good())
		{
			cout<<"Saved dispersion at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dispersion at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	// JROT2
	if(which.jrot2)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Jrot2.txt";		
		output.open(temp.str(),fstream::app);
		double *jrot2=new double[which.bins];
		obs->write_jrot2(jrot2);
		for(int i=0; i<which.bins; i++)
		{
			output<<jrot2[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved jrot2 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving jrot2 at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/Jrot2_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *jrot2=new double[which.bins];
		obs->write_jrot2(jrot2);
		for(int i=0; i<which.bins; i++)
		{
			output<<jrot2[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved jrot2 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving jrot2 at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	// DRIFT 
	if(which.drift)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Drift.txt";		
		output.open(temp.str(),fstream::app);
		int *drift=new int[which.drift_bins];
		obs->write_drift(drift);
		for(int i=0; i<which.drift_bins; i++)
		{
			output<<drift[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved drift distribution at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving drift distribution at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/Drift_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		int *drift=new int[which.drift_bins];
		obs->write_drift(drift);
		for(int i=0; i<which.drift_bins; i++)
		{
			output<<drift[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved drift distribution at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving drift distribution at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}
}

void SaveResults::save_lattice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/Lattice_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);
	
	for(int i=0; i < lattice->get_length(); i++)
	{
		output<<data1[i]<<endl;
		output<<data2[i]<<endl;
	}
	
	output.close();
}

void SaveResults::save_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int tauindex)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/Lattice_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);
	
	for(int compindex=0; compindex<COMPONENTS; compindex++)
	{
		for(int xindex=0; xindex < XSIZE; xindex++)
		{
			for(int yindex=0; yindex < YSIZE; yindex++)
			{
				for(int zindex=0; zindex < ZSIZE; zindex++)
				{
						output<<data1[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
						output<<data2[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
				}
			}
		}
	}
	
	if(output.good())
	{
		cout<<"Saved lattice at time "<<number<<endl;
	}
	else
	{
		cout<<"Saving lattice at time "<<number<<" failed"<<endl;
	}
	output.close();
}

void SaveResults::save_lattice_slices(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int min_tauindex, int max_tauindex)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/Lattice_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);
	
	for(int compindex=0; compindex<COMPONENTS; compindex++)
	{
	    for(int tauindex=min_tauindex; tauindex<max_tauindex; tauindex++)
		{
		for(int xindex=0; xindex < XSIZE; xindex++)
		{
			for(int yindex=0; yindex < YSIZE; yindex++)
			{
				for(int zindex=0; zindex < ZSIZE; zindex++)
				{
						output<<data1[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
						output<<data2[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
				}
			}
		}
		}
	}
	
	if(output.good())
	{
		cout<<"Saved lattice at time "<<number<<endl;
	}
	else
	{
		cout<<"Saving lattice at time "<<number<<" failed"<<endl;
	}
	output.close();
}



void SaveResults::read_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, string filename)
{
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream input;		
	input.open(filename,fstream::in);
	complex <double> temp1,temp2;
	
	for(int compindex=0; compindex<COMPONENTS; compindex++)
	{
		for(int xindex=0; xindex < XSIZE; xindex++)
		{
			for(int yindex=0; yindex < YSIZE; yindex++)
			{
				for(int zindex=0; zindex < ZSIZE; zindex++)
				{
						input>>temp1; input>>temp2;
						for(int tauindex=0; tauindex < TAUSIZE; tauindex++)
						{
							data1[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]=temp1;
							data2[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]=temp2;
						}
					
				}
			}
		}
	}
	input.close();
}
