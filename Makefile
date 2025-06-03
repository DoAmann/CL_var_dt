CUDA_LIB_DIR = /opt/spack_views/ido/targets/x86_64-linux/lib
LD_LIBRARY_PATH=$(CUDA_LIB_DIR)
all: 2D_bosegascl2
2D_bosegascl2: main.cu Parameters.h Lattice.cu Propagator_without_cutoff.cu Observables_without_cutoff.cu SaveResults.cu
	nvcc -arch=sm_80 -Wno-deprecated-gpu-targets -L$(CUDA_LIB_DIR) --linker-options -rpath,$(LD_LIBRARY_PATH) -lcurand -lcufft -o  2D_bosegascl2 main.cu
