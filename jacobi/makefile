#
# Copyright (C) 2015-2016 Kalray SA. All rights reserved.
#

K1_TOOLCHAIN_DIR=/usr/local/k1tools
system-name := bare

# Directories.
BINDIR  = $(CURDIR)/bin
INCDIR  = $(CURDIR)/../../include
SRCDIR  = $(CURDIR)

# Toolchain Configuration
cppflags := -std=c++14
cppflags += -Wall
cppflags += -O3
cppflags += -I $(INCDIR)
cppflags += -D_KALRAY_MPPA256
lflags := -Wl,--defsym=_LIBNOC_DISABLE_FIFO_FULL_CHECK=0

#=============================================================================
# Game of Life - Async API
#=============================================================================

io-bin += jacobi-async-master
jacobi-async-master-srcs := $(SRCDIR)/master.cpp
 
# Toolchain Configuration
jacobi-async-master-system := bare
jacobi-async-master-lflags := -mhypervisor -lutask -lmppa_async -lmppa_request_engine
jacobi-async-master-lflags += -lmppapower -lmppanoc -lmpparouting
jacobi-async-master-lflags += -lpcie_queue


cluster-bin += jacobi-async-slave
jacobi-async-slave-srcs := $(SRCDIR)/slave.cpp       \
                        $(INCDIR)/mppa_utils.cpp 

# Toolchain Configuration
jacobi-async-slave-system   := bare
jacobi-async-slave-cppflags += -fopenmp
jacobi-async-slave-lflags   := -mhypervisor -lutask -lmppa_async -lmppa_request_engine 
jacobi-async-slave-lflags   += -lmppapower -lmppanoc -lmpparouting -lgomp
jacobi-async-slave-lflags   += -Wl,--defsym=USER_STACK_SIZE=0x2000
jacobi-async-slave-lflags   += -Wl,--defsym=KSTACK_SIZE=0x1000


jacobi-async-objs := jacobi-async-master jacobi-async-slave
jacobi-async-name := jacobi-async.img

mppa-bin := jacobi-async

include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.kalray