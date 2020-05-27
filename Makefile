NVCC = $(shell which nvcc)
CC = g++-9
OPT = -g #-O2 #-g
NVCC_FLAGS = $(OPT) -Xcompiler -fopenmp -I /opt/eigen-3.3.7/ -I /usr/include/crt
CC_FLAGS = $(OPT) -fopenmp -Wall -I /usr/local/cuda/include -I /opt/eigen-3.3.7/

omp_rotate = xomp_rotate.x
comp_omp_flip = xc_omp_rotate.x
cfomp_rotate = xcf_omp_rotate.x
cbomp_rotate = xcb_omp_rotate.x
bomp_rotate = xb_omp_rotate.x
gbomp_rotate =xgb_omp_rotate.x
edgeG1 = xcudaEdge.x

all: $(omp_rotate) $(comp_omp_flip) $(cfomp_rotate) $(cbomp_rotate) $(bomp_rotate) $(gbomp_rotate) $(edgeG1)

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

$(cbomp_rotate): drv_cb_omp_rotate.o cb_omp_rotate.o imageBMP.o
	@echo "----- Building $(cbomp_rotate) -----"
	$(CC) $(CC_FLAGS) -fopenmp $^ -lpthread -o $@
	@echo

$(bomp_rotate): drv_b_omp_rotate.o b_omp_rotate.o imageBMP.o
	@echo "----- Building $(bomp_rotate) -----"
	$(CC) $(CC_FLAGS) -fopenmp $^ -lpthread -o $@
	@echo

$(gbomp_rotate): drv_gb_omp_rotate.o gb_omp_rotate.o imageBMP.o
	@echo "----- Building $(gbomp_rotate) -----"
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo

$(edgeG1): drv_cuda_edge.o cuda_edge.o imageBMP.o
	@echo "----- Building $(edgeG1) -----"
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
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
