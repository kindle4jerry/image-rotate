NVCC = $(shell which nvcc)
CC = g++-9
OPT = -g #-O2 #-g
NVCC_FLAGS = $(OPT) -G -Xcompiler -Wall
CC_FLAGS = $(OPT) -fopenmp -Wall

omp_rotate = xomp_rorate.x
comp_omp_flip = xc_omp_rorate.x
cfomp_rotate = xcf_omp_rorate.x

all: $(omp_rotate) $(comp_omp_flip) $(cfomp_rotate)

$(omp_rotate): drv_omp_rotate.o omp_rotate.o imageBMP.o
	@echo "----- Building $(omp_rotate) -----"
	$(CC) $(CC_FLAGS) -fopenmp $^ -lpthread -o $@
	@echo

$(comp_omp_flip): drv_c_omp_rotate.o c_omp_rotate.o imageBMP.o
	@echo "----- Building $(comp_omp_rotate) -----"
	$(CC) $(CC_FLAGS) -fopenmp $^ -lpthread -o $@
	@echo

$(cfomp_rotate): drv_cf_omp_rotate.o cf_omp_rotate.o imageBMP.o
	@echo "----- Building $(cfomp_rotate) -----"
	$(CC) $(CC_FLAGS) -fopenmp $^ -lpthread -o $@
	@echo

$(mem_omp_flip_for): drv_memOmp_imrotate_for.o memOmpRotate_for.o imageBMP.o
	$(CC) $(CC_FLAGS) -fopenmp $^ -lpthread -o $@


%.o: %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@ 
%.o: %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@
%.o : %.cu 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.x
