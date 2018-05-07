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

io-bin += gol-async-master
gol-async-master-srcs := $(SRCDIR)/master.cpp
 
# Toolchain Configuration
gol-async-master-system := bare
gol-async-master-lflags := -mhypervisor -lutask -lmppa_async -lmppa_request_engine
gol-async-master-lflags += -lmppapower -lmppanoc -lmpparouting
gol-async-master-lflags += -lpcie_queue


cluster-bin += gol-async-slave
gol-async-slave-srcs := $(SRCDIR)/slave.cpp       \
                        $(SRCDIR)/mppa_utils.cpp 

# Toolchain Configuration
gol-async-slave-system   := bare
gol-async-slave-cppflags += -fopenmp
gol-async-slave-lflags   := -mhypervisor -lutask -lmppa_async -lmppa_request_engine 
gol-async-slave-lflags   += -lmppapower -lmppanoc -lmpparouting -lgomp
gol-async-slave-lflags   += -Wl,--defsym=USER_STACK_SIZE=0x2000
gol-async-slave-lflags   += -Wl,--defsym=KSTACK_SIZE=0x1000


gol-async-objs := gol-async-master gol-async-slave
gol-async-name := gol-async.img

mppa-bin := gol-async

include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.kalray