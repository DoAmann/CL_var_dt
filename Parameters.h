#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//physical parameters

#define COMPONENTS 4

#define XSIZE 256
#define YSIZE 1
#define ZSIZE 1
#define TAUSIZE 32

#define EPS     0.01		    //beta=EPS*TAUSIZE, T=1/beta (not make it larger) EPS=0.025
#define a_x     1.0             //lattice spacing in x direction
#define G       0.10000         //coupling constant
#define M       0.00000		        //dipole coupling (C_dd)
#define SIGMA   0.00000         //tilting angle
#define D       0.27735         //sigma of the z gaussian solution d^2=1/(m*omega_z)

#define OMEGAX 0.		//strength of trapping potential in x direction, OMEGAX=0.5*m*omega_x^2
#define OMEGAY 0.		//strength of trapping potential in y direction, OMEGAY=0.5*m*omega_y^2
#define OMEGAZ 0.		//strength of trapping potential in z direction, OMEGAZ=0.5*m*omega_z^2
#define Q 0.0             //strength of quadratic Zeeman effect   



//numerical parameters
#define STEPS 8000000
#define DT 0.002         //2.e-3
#define SNAPSHOT_TRIGGER 500000
#define EVALUATE_TRIGGER 100


//computational parameters
#define BLOCKSIZE 64		//block size for calculations with one-dimensional grid (e. g. summation of lattices)
	
//Block sizes for calculations with four-dimensional grid. Must be divisors of the lattice dimensions. If shared memory is used, choose as cubic as possible. 
 
#define BLOCKSIZEX 16		
#define BLOCKSIZEY 1
#define BLOCKSIZEZ 1
#define BLOCKSIZETAU 16


#define TERM_EXP 1.961160657620889258279106570626026950776576995849609375000000e-01
#define PREFACTOR 0.000000000000000000000000000000000000000000000000000000000000e+00
#define TERM1 1.042820027953361927686160015582572668790817260742187500000000e+00
#define SIN2 0.000000000000000000000000000000000000000000000000000000000000e+00
#define COS2 1.000000000000000000000000000000000000000000000000000000000000e+00
#define G_2D 7.192036783872953753515844255161937326192855834960937500000000e-01

#endif

