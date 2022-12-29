
#
# a simple executable
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_target(EXEC
    NAME exec
    CU_SOURCES exec.cpp
    CU_ARCH "${CU_ARCS}"
    LINK_LIBS public submod)

