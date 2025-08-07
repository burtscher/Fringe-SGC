CXX = g++
NVCC = nvcc

CXXFLAGS = -O3 -fopenmp
NVCCFLAGS = -O3 -arch=sm_86


SRCDIR = src

#source files
CPPS = $(wildcard $(SRCDIR)/*.cpp)
CUS  = $(wildcard $(SRCDIR)/*.cu)

CPP_TARGETS = $(patsubst $(SRCDIR)/%.cpp, $(SRCDIR)/%, $(CPPS))
CU_TARGETS  = $(patsubst $(SRCDIR)/%.cu, $(SRCDIR)/%, $(CUS))

$(info CPP files: $(CPPS))
$(info CU files: $(CUS))

all: $(CPP_TARGETS) $(CU_TARGETS) setup

$(SRCDIR)/%: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(SRCDIR)/%: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

setup:
	bash setup.sh
