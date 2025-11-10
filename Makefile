# ---- config ---------------------------------------------------------
CXX      ?= g++
CXXFLAGS ?= -g -std=c++17 -Iinclude -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?=

SRCDIR   := src
INCDIR   := include
OBJDIR   := build
BINDIR   := bin

ifeq ($(OS),Windows_NT)
  EXE      := .exe
  MKDIR_P  := mkdir
  RM_RF    := rmdir /S /Q
  DEVNULL  := NUL
  LDFLAGS  += -lpsapi
else
  EXE      :=
  MKDIR_P  := mkdir -p
  RM_RF    := rm -rf
  DEVNULL  := /dev/null
endif

ALGO_SRCS := whitman.cpp freese.cpp cosmadakis.cpp hunt.cpp
RUNNER    := runner_min.cpp
GEN_SRC   := lattice_gen.cpp

ALGO_OBJS := $(addprefix $(OBJDIR)/, $(ALGO_SRCS:.cpp=.o))
RUNNER_OBJ:= $(OBJDIR)/$(RUNNER:.cpp=.o)

all: $(BINDIR)/check$(EXE) $(BINDIR)/lattice_gen$(EXE)

$(BINDIR)/check$(EXE): $(ALGO_OBJS) $(RUNNER_OBJ) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/lattice_gen$(EXE): $(SRCDIR)/$(GEN_SRC) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR) $(BINDIR):
	-@$(MKDIR_P) $@ 2>$(DEVNULL)

clean:
	-$(RM_RF) $(OBJDIR)
	-$(RM_RF) $(BINDIR)

.PHONY: all clean
