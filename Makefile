OS           := $(shell uname)
BUILDDIR     := ./build
SRCDIR       := ./src
CACHE_DIR    := ./.cache
NVCC         := nvcc
CUDA_INCLUDE := /opt/cuda/include # Default installation path for ArchLinux, may be different

NVCC_INSTALLED := $(shell command -v $(NVCC) 2> /dev/null)

COMMON_SRCDIR := $(SRCDIR)/common
DENSE_SRCDIR  := $(SRCDIR)/dense
CNN_SRCDIR    := $(SRCDIR)/cnn

DENSE_SRC    := $(wildcard $(DENSE_SRCDIR)/*.c)
CNN_SRC      := $(wildcard $(CNN_SRCDIR)/*.c)
CNN_SRC_CUDA := $(wildcard $(CNN_SRCDIR)/*.cu)

DENSE_OBJ     = $(filter-out $(BUILDDIR)/dense_main.o $(BUILDDIR)/dense_utils.o $(BUILDDIR)/dense_preview.o, $(DENSE_SRC:$(DENSE_SRCDIR)/%.c=$(BUILDDIR)/dense_%.o))
CNN_OBJ       = $(filter-out $(BUILDDIR)/cnn_main.o $(BUILDDIR)/cnn_preview.o $(BUILDDIR)/cnn_export.o, $(CNN_SRC:$(CNN_SRCDIR)/%.c=$(BUILDDIR)/cnn_%.o))
CNN_OBJ_CUDA  = $(CNN_SRC:$(CNN_SRCDIR)/%.cu=$(BUILDDIR)/cnn_%.o)


TEST_SRCDIR  := test

TESTS_SRC     = $(wildcard $(TEST_SRCDIR)/*.c)
TESTS_SRC_CU += $(wildcard $(TEST_SRCDIR)/*.cu)

TESTS_OBJ     = $(TESTS_SRC:$(TEST_SRCDIR)/%.c=$(BUILDDIR)/$(TEST_SRCDIR)-%) $(TESTS_SRC_CU:$(TEST_SRCDIR)/%.cu=$(BUILDDIR)/$(TEST_SRCDIR)-%)

# Linker only flags
LD_CFLAGS    =  -lm -lpthread -ljpeg
LD_NVCCFLAGS = -ljpeg

# Compilation flag
CFLAGS    = -Wall -Wextra -std=gnu99 -g -O3
NVCCFLAGS = -g
# Remove warnings about unused variables, functions, ...
# -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable
# Compile with debug
# -g
# See memory leaks and Incorrect Read/Write
# -fsanitize=address -lasan
#! WARNING: test/cnn-neuron_io fails with this option enabled

# Specify library path of libjpeg on MacOS
ifeq ($(OS),Darwin)
	LD_CFLAGS    += -I/opt/homebrew/Cellar/jpeg/9e/include/ -L/opt/homebrew/Cellar/jpeg/9e/lib/
	LD_NVCCFLAGS += -L/opt/homebrew/Cellar/jpeg/9e/lib/

	CFLAGS       +=  -I/opt/homebrew/Cellar/jpeg/9e/include/
endif


all: dense cnn;



#
# Build dense
#
# Executables
dense: $(BUILDDIR)/dense-main $(BUILDDIR)/dense-utils $(BUILDDIR)/dense-preview;

$(BUILDDIR)/dense-main: $(DENSE_SRCDIR)/main.c $(BUILDDIR)/mnist.o $(BUILDDIR)/dense_neuron_io.o $(BUILDDIR)/dense_neural_network.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)

$(BUILDDIR)/dense-utils: $(DENSE_SRCDIR)/utils.c $(BUILDDIR)/dense_neural_network.o $(BUILDDIR)/dense_neuron_io.o $(BUILDDIR)/mnist.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)

$(BUILDDIR)/dense-preview: $(DENSE_SRCDIR)/preview.c $(BUILDDIR)/mnist.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)

$(BUILDDIR)/dense_%.o: $(DENSE_SRCDIR)/%.c $(DENSE_SRCDIR)/include/%.h
	$(CC)  -c $< -o $@  $(CFLAGS)



#
# Build cnn
#
cnn: $(BUILDDIR)/cnn-main $(BUILDDIR)/cnn-main-cuda $(BUILDDIR)/cnn-preview $(BUILDDIR)/cnn-export $(BUILDDIR)/cnn-export-cuda;

$(BUILDDIR)/cnn-main: $(CNN_SRCDIR)/main.c \
		$(BUILDDIR)/cnn_backpropagation.o \
		$(BUILDDIR)/cnn_initialisation.o \
		$(BUILDDIR)/cnn_test_network.o \
		$(BUILDDIR)/cnn_convolution.o \
		$(BUILDDIR)/cnn_neuron_io.o \
		$(BUILDDIR)/cnn_function.o  \
		$(BUILDDIR)/cnn_creation.o \
		$(BUILDDIR)/cnn_models.o \
		$(BUILDDIR)/cnn_update.o \
		$(BUILDDIR)/cnn_train.o \
		$(BUILDDIR)/cnn_utils.o \
		$(BUILDDIR)/cnn_make.o \
		$(BUILDDIR)/cnn_free.o \
		$(BUILDDIR)/cnn_jpeg.o \
		$(BUILDDIR)/cnn_cnn.o \
		\
		$(BUILDDIR)/memory_management.o \
		$(BUILDDIR)/colors.o \
		$(BUILDDIR)/mnist.o \
		$(BUILDDIR)/utils.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)


ifdef NVCC_INSTALLED
$(BUILDDIR)/cnn-main-cuda: $(BUILDDIR)/cnn_main.cuda.o \
		$(BUILDDIR)/cnn_initialisation.cuda.o \
		$(BUILDDIR)/cnn_test_network.cuda.o \
		$(BUILDDIR)/cnn_neuron_io.cuda.o \
		$(BUILDDIR)/cnn_creation.cuda.o \
		$(BUILDDIR)/cnn_models.cuda.o \
		$(BUILDDIR)/cnn_update.cuda.o \
		$(BUILDDIR)/cnn_train.cuda.o \
		$(BUILDDIR)/cnn_utils.cuda.o \
		$(BUILDDIR)/cnn_free.cuda.o \
		$(BUILDDIR)/cnn_jpeg.cuda.o \
		$(BUILDDIR)/cnn_cnn.cuda.o \
		\
		$(BUILDDIR)/cnn_cuda_backpropagation.o \
		$(BUILDDIR)/cnn_cuda_convolution.o \
		$(BUILDDIR)/cnn_cuda_function.o  \
		$(BUILDDIR)/cnn_cuda_make.o \
		\
		$(BUILDDIR)/cuda_memory_management.o \
		$(BUILDDIR)/colors.cuda.o \
		$(BUILDDIR)/mnist.cuda.o \
		$(BUILDDIR)/cuda_utils.o		
	$(NVCC)  $(LD_NVCCFLAGS) $(NVCCFLAGS)  $^ -o $@
else
$(BUILDDIR)/cnn-main-cuda:
	@echo "$(NVCC) not found, skipping"
endif


$(BUILDDIR)/cnn-preview: $(CNN_SRCDIR)/preview.c $(BUILDDIR)/cnn_jpeg.o $(BUILDDIR)/colors.o $(BUILDDIR)/utils.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)


$(BUILDDIR)/cnn-export: $(CNN_SRCDIR)/export.c \
		$(BUILDDIR)/cnn_backpropagation.o \
		$(BUILDDIR)/cnn_convolution.o \
		$(BUILDDIR)/cnn_neuron_io.o \
		$(BUILDDIR)/cnn_function.o \
		$(BUILDDIR)/cnn_free.o \
		$(BUILDDIR)/cnn_make.o \
		$(BUILDDIR)/cnn_cnn.o \
		$(BUILDDIR)/cnn_jpeg.o \
		\
		$(BUILDDIR)/memory_management.o \
		$(BUILDDIR)/colors.o \
		$(BUILDDIR)/mnist.o \
		$(BUILDDIR)/utils.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)


ifdef NVCC_INSTALLED
$(BUILDDIR)/cnn-export-cuda: $(CNN_SRCDIR)/export.c \
		$(BUILDDIR)/cnn_cuda_backpropagation.o \
		$(BUILDDIR)/cnn_cuda_convolution.o \
		$(BUILDDIR)/cnn_neuron_io.cuda.o \
		$(BUILDDIR)/cnn_cuda_function.o \
		$(BUILDDIR)/cnn_free.cuda.o \
		$(BUILDDIR)/cnn_cuda_make.o \
		$(BUILDDIR)/cnn_cnn.cuda.o \
		$(BUILDDIR)/cnn_utils.cuda.o \
		$(BUILDDIR)/cnn_jpeg.cuda.o \
		\
		$(BUILDDIR)/cuda_memory_management.o \
		$(BUILDDIR)/colors.cuda.o \
		$(BUILDDIR)/mnist.o \
		$(BUILDDIR)/cuda_utils.o
	$(NVCC)  $^ -o $@  $(NVCCFLAGS) $(LD_NVCCFLAGS)
else
$(BUILDDIR)/cnn-export-cuda: $(CNN_SRCDIR)/export.c
	@echo "$(NVCC) not found, skipping"
endif

$(BUILDDIR)/cnn_%.o: $(CNN_SRCDIR)/%.c $(CNN_SRCDIR)/include/%.h
	$(CC)  -c $< -o $@  $(CFLAGS)


$(BUILDDIR)/cnn_%.cuda.o: $(CNN_SRCDIR)/%.c $(CNN_SRCDIR)/include/%.h
	$(CC)  -c $< -o $@  $(CFLAGS) -DUSE_CUDA -lcuda -I$(CUDA_INCLUDE)


ifdef NVCC_INSTALLED
$(BUILDDIR)/cnn_cuda_%.o: $(CNN_SRCDIR)/%.cu $(CNN_SRCDIR)/include/%.h
	$(NVCC)  $(NVCCFLAGS)  -c -dc $< -o $@
else
$(BUILDDIR)/cnn_cuda_%.o: $(CNN_SRCDIR)/%.cu $(CNN_SRCDIR)/include/%.h
	@echo "$(NVCC) not found, skipping"
endif



#
# Build general files
#
$(BUILDDIR)/%.o: $(COMMON_SRCDIR)/%.c $(COMMON_SRCDIR)/include/%.h
	$(CC)  -c $< -o $@  $(CFLAGS)


$(BUILDDIR)/%.cuda.o: $(COMMON_SRCDIR)/%.c $(COMMON_SRCDIR)/include/%.h
	$(CC)  -c $< -o $@  $(CFLAGS) -DUSE_CUDA -lcuda -I$(CUDA_INCLUDE)


ifdef NVCC_INSTALLED
$(BUILDDIR)/cuda_%.o: $(COMMON_SRCDIR)/%.cu $(COMMON_SRCDIR)/include/%.h
	$(NVCC)  $(NVCCFLAGS)  -c -dc $< -o $@
else
	@echo "$(NVCC) not found, skipping"
endif



#
# Tests
#
run-tests: build-tests
	$(foreach file, $(wildcard $(TEST_SRCDIR)/*.sh), $(file);)
	@echo "$$(for file in build/test-*; do $$file || exit 1; done)"

build-tests: prepare-tests $(TESTS_OBJ) $(BUILDDIR)/test-cnn_matrix_multiplication $(BUILDDIR)/test-cnn_convolution $(BUILDDIR)/test-cuda_memory_management


prepare-tests:
	@rm -f $(BUILDDIR)/test-*


$(BUILDDIR)/test-cnn_%: $(TEST_SRCDIR)/cnn_%.c $(CNN_OBJ) $(BUILDDIR)/colors.o $(BUILDDIR)/mnist.o $(BUILDDIR)/utils.o $(BUILDDIR)/memory_management.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)

$(BUILDDIR)/test-dense_%: $(TEST_SRCDIR)/dense_%.c $(DENSE_OBJ) $(BUILDDIR)/colors.o $(BUILDDIR)/mnist.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)

$(BUILDDIR)/test-memory_management: $(TEST_SRCDIR)/memory_management.c $(BUILDDIR)/colors.o $(BUILDDIR)/utils.o $(BUILDDIR)/test_memory_management.o
	$(CC)  $^ -o $@  $(CFLAGS) $(LD_CFLAGS)

$(BUILDDIR)/test_memory_management.o: $(COMMON_SRCDIR)/memory_management.c $(COMMON_SRCDIR)/include/memory_management.h
	$(CC)  -c $< -o $@  $(CFLAGS) -DTEST_MEMORY_MANAGEMENT

ifdef NVCC_INSTALLED
$(BUILDDIR)/test-cuda_memory_management: $(TEST_SRCDIR)/memory_management.cu $(BUILDDIR)/colors.cuda.o $(BUILDDIR)/cuda_utils.o $(BUILDDIR)/cuda_memory_management.o
	$(NVCC)  $(LD_NVCCFLAGS) $(NVCCFLAGS)  $^ -o $@
else
$(BUILDDIR)/test-cuda_memory_management:
	@echo "$(NVCC) not found, skipping"
endif

ifdef NVCC_INSTALLED
$(BUILDDIR)/test-cnn_%: $(TEST_SRCDIR)/cnn_%.cu \
		$(BUILDDIR)/cnn_cuda_%.o \
		$(BUILDDIR)/cuda_utils.o \
		$(BUILDDIR)/colors.o \
		$(BUILDDIR)/mnist.cuda.o \
		$(BUILDDIR)/cuda_memory_management.o
	$(NVCC)  $(LD_NVCCFLAGS) $(NVCCFLAGS)  $^ -o $@
else
$(BUILDDIR)/test-cnn_%: $(TEST_SRCDIR)/cnn_%.cu
	@echo "$(NVCC) not found, skipping"
endif



#
# Utils
#
webserver: $(CACHE_DIR)/mnist-reseau-fully-connected.bin $(CACHE_DIR)/mnist-reseau-cnn.bin
	FLASK_APP="src/webserver/app.py" flask run

$(CACHE_DIR)/mnist-reseau-fully-connected.bin: $(BUILDDIR)/dense-main
	@mkdir -p $(CACHE_DIR)
	$(BUILDDIR)/dense-main train \
		--images "data/mnist/train-images-idx3-ubyte" \
		--labels "data/mnist/train-labels-idx1-ubyte" \
		--out "$(CACHE_DIR)/mnist-reseau-fully-connected.bin"


$(CACHE_DIR)/mnist-reseau-cnn.bin: $(BUILDDIR)/cnn-main
	@mkdir -p $(CACHE_DIR)
	$(BUILDDIR)/cnn-main train \
		--dataset mnist \
		--images data/mnist/train-images-idx3-ubyte \
		--labels data/mnist/train-labels-idx1-ubyte \
		--epochs 10 \
		--out $(CACHE_DIR)/mnist-reseau-cnn.bin 



#
# Clean project
#
clean:
	rm -rf $(BUILDDIR)/*

#rm -f $(CACHE_DIR)/*
