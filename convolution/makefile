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

io-bin += convolution-async-master
convolution-async-master-srcs := $(SRCDIR)/master.cpp
 
# Toolchain Configuration
convolution-async-master-system := bare
convolution-async-master-lflags := -mhypervisor -lutask -lmppa_async -lmppa_request_engine
convolution-async-master-lflags += -lmppapower -lmppanoc -lmpparouting
convolution-async-master-lflags += -lpcie_queue


cluster-bin += convolution-async-slave
convolution-async-slave-srcs := $(SRCDIR)/slave.cpp       \
                        $(INCDIR)/mppa_utils.cpp 

# Toolchain Configuration
convolution-async-slave-system   := bare
convolution-async-slave-cppflags += -fopenmp
convolution-async-slave-lflags   := -mhypervisor -lutask -lmppa_async -lmppa_request_engine 
convolution-async-slave-lflags   += -lmppapower -lmppanoc -lmpparouting -lgomp
convolution-async-slave-lflags   += -Wl,--defsym=USER_STACK_SIZE=0x2000
convolution-async-slave-lflags   += -Wl,--defsym=KSTACK_SIZE=0x1000


convolution-async-objs := convolution-async-master convolution-async-slave
convolution-async-name := convolution-async.img

mppa-bin := convolution-async

include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.kalray