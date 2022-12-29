set(INCLUDE_PREFIX include)
set(LIB_PREFIX lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# core: this is the core library with the bulk of our code. better if static
#
add_target(LIB
    NAME core
    CU_SOURCES add.cpp
    CU_ARCH "${CU_ARCS}"
    CXX_SOURCES add_cpu.cpp
    LINK_LIBS submod)

target_include_directories(core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>)

#
# extension: an extension that depends on and is required to be shared library
#
add_target(SHARED_LIB
    NAME extension
    CU_SOURCES mul.cpp
    CU_ARCH "${CU_ARCS}"
    CXX_SOURCES mul_cpu.cpp
    LINK_LIBS submod core)

target_include_directories(extension INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>)

#
# public: the interfaces to both core and extension libraries. better if static
#
add_target(LIB
    NAME public
    CXX_SOURCES div.cpp
    LINK_LIBS core extension)

target_include_directories(public INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>)

