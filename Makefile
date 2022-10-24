BUILDDIR     := ./build
SRCDIR       := ./src
CACHE_DIR    := ./cache
NVCC         := nvcc
CC           := gcc

NVCC_INSTALLED := $(shell command -v nvcc 2> /dev/null)

MNIST_SRCDIR := $(SRCDIR)/mnist
CNN_SRCDIR   := $(SRCDIR)/cnn

MNIST_SRC    := $(wildcard $(MNIST_SRCDIR)/*.c)
CNN_SRC      := $(wildcard $(CNN_SRCDIR)/*.c)
CNN_SRC_CUDA := $(wildcard $(CNN_SRCDIR)/*.cu)

MNIST_OBJ     = $(filter-out $(BUILDDIR)/mnist_main.o $(BUILDDIR)/mnist_utils.o $(BUILDDIR)/mnist_preview.o, $(MNIST_SRC:$(MNIST_SRCDIR)/%.c=$(BUILDDIR)/mnist_%.o))
CNN_OBJ       = $(filter-out $(BUILDDIR)/cnn_main.o, $(CNN_SRC:$(CNN_SRCDIR)/%.c=$(BUILDDIR)/cnn_%.o))
CNN_OBJ_CUDA  = $(CNN_SRC:$(CNN_SRCDIR)/%.cu=$(BUILDDIR)/cnn_%.o)


TEST_SRCDIR  := test

TESTS_SRC     = $(wildcard test/*.c)
TESTS_SRC_CU += $(wildcard test/*.cu)

TESTS_OBJ     = $(TESTS_SRC:test/%.c=$(BUILDDIR)/test-%) $(TESTS_SRC_CU:test/%.cu=$(BUILDDIR)/test-%)

# Compile flags
CFLAGS   = -std=c99 -lm -lpthread
NVCCFLAGS = 

# Additional warning rules
CFLAGS   += -Wall -Wextra
NVCCFLAGS +=
# Remove warnings about unused variables, functions, ...
# -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable
# Compile with debug
# -g

all: mnist cnn;
#
# Build mnist
#
# Executables
mnist: $(BUILDDIR)/mnist-main $(BUILDDIR)/mnist-utils $(BUILDDIR)/mnist-preview;

$(BUILDDIR)/mnist-main: $(MNIST_SRCDIR)/main.c $(BUILDDIR)/mnist.o $(BUILDDIR)/mnist_neuron_io.o $(BUILDDIR)/mnist_neural_network.o
	$(CC)  $(CFLAGS)  $(MNIST_SRCDIR)/main.c $(BUILDDIR)/mnist.o $(BUILDDIR)/mnist_neuron_io.o $(BUILDDIR)/mnist_neural_network.o -o $(BUILDDIR)/mnist-main

$(BUILDDIR)/mnist-utils: $(MNIST_SRCDIR)/utils.c $(BUILDDIR)/mnist_neural_network.o $(BUILDDIR)/mnist_neuron_io.o $(BUILDDIR)/mnist.o
	$(CC)  $(CFLAGS)  $^ -o $@

$(BUILDDIR)/mnist-preview: $(MNIST_SRCDIR)/preview.c $(BUILDDIR)/mnist.o
	$(CC)  $(CFLAGS)  $^ -o $@

# .o files
$(BUILDDIR)/mnist.o: $(MNIST_SRCDIR)/mnist.c $(MNIST_SRCDIR)/include/mnist.h
	$(CC)  $(CFLAGS)  -c $< -o $@

$(BUILDDIR)/mnist_%.o: $(MNIST_SRCDIR)/%.c $(MNIST_SRCDIR)/include/%.h
	$(CC)  $(CFLAGS)  -c $< -o $@


#
# Build cnn
#
cnn: $(BUILDDIR)/cnn-main;

$(BUILDDIR)/cnn-main: $(CNN_SRCDIR)/main.c $(BUILDDIR)/cnn_train.o $(BUILDDIR)/cnn_cnn.o $(BUILDDIR)/cnn_creation.o $(BUILDDIR)/cnn_initialisation.o $(BUILDDIR)/cnn_make.o $(BUILDDIR)/cnn_neuron_io.o $(BUILDDIR)/cnn_function.o  $(BUILDDIR)/cnn_utils.o $(BUILDDIR)/cnn_free.o $(BUILDDIR)/colors.o $(BUILDDIR)/mnist.o
	$(CC)  $(CFLAGS)  $^ -o $@

$(BUILDDIR)/cnn_%.o: $(CNN_SRCDIR)/%.c $(CNN_SRCDIR)/include/%.h
	$(CC)  $(CFLAGS)  -c $< -o $@

$(BUILDDIR)/cnn_%.o: $(CNN_SRCDIR)/%.cu $(CNN_SRCDIR)/include/%.h
ifndef NVCC_INSTALLED
	@echo "nvcc not found, skipping"
else
	$(NVCC)  $(NVCCFLAGS)  -c $< -o $@
endif
#
# Build general files
#
$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/include/%.h
	$(CC)  $(CFLAGS)  -c $< -o $@

#
# Tests
#
run-tests: build-tests
	$(foreach file, $(wildcard $(BUILDDIR)/test-*), $(file);)
	$(foreach file, $(wildcard $(TEST_SRCDIR)/*.sh), $(file);)

build-tests: prepare-tests $(TESTS_OBJ)


prepare-tests:
	@rm -f $(BUILDDIR)/test-*


build/test-cnn_%: test/cnn_%.c $(CNN_OBJ) $(BUILDDIR)/colors.o $(BUILDDIR)/mnist.o
	$(CC)  $(CFLAGS)  $^ -o $@

# mnist.o est déjà inclus en tant que mnist_mnist.o
build/test-mnist_%: test/mnist_%.c $(MNIST_OBJ) $(BUILDDIR)/colors.o
	$(CC)  $(CFLAGS)  $^ -o $@

$(BUILDDIR)/test-cnn_matrix_multiplication: test/cnn_matrix_multiplication.cu $(BUILDDIR)/cnn_matrix_multiplication.o $(BUILDDIR)/colors.o $(BUILDDIR)/mnist.o
ifndef NVCC_INSTALLED
	@echo "nvcc not found, skipping"
else
	$(NVCC)  $(NVCCFLAGS)  $^ -o $@
endif

#
# Utils
#
webserver: $(CACHE_DIR)/mnist-reseau.bin
	FLASK_APP="src/webserver/app.py" flask run

$(CACHE_DIR)/mnist-reseau.bin: $(BUILDDIR)/mnist-main
	@mkdir -p $(CACHE_DIR)
	$(BUILDDIR)/mnist-main train \
		--images "data/mnist/train-images-idx3-ubyte" \
		--labels "data/mnist/train-labels-idx1-ubyte" \
		--out "$(CACHE_DIR)/mnist-reseau.bin"


#
# Clean project
#
clean:
	rm -rf $(BUILDDIR)/*
	rm -f $(CACHE_DIR)/*