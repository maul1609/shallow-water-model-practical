CXX = g++
NVCC = nvcc
RM = rm -rf

all: shallow_water_model_cuda

shallow_water_model_cuda: main.cu
	$(NVCC) -o shallow_water_model_cuda $<

clean:
	$(RM) shallow_water_model_cuda $(OBJECTS)
