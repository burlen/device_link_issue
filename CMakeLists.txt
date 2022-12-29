# only require the default version in the oldest active ubuntu lts release
cmake_minimum_required(VERSION 3.6)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0127 NEW)
cmake_policy(SET CMP0105 NEW)

project(debug_linking)

#
# compile CUDA code for these GPU's
#
set(CU_ARCS "60;70;75" CACHE STRING "cuda architectures")

#
# global compilation options
#
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_ARCHITECTURES ${CU_ARCS})
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
enable_language(CUDA)



#
# a function to configure targets consistently and report in the terminal=
#
function(add_target)
  set(OPTS LIB SHARED_LIB STATIC_LIB EXEC CUDA)
  set(NVPO NAME CU_ARCH)
  set(MVO CXX_SOURCES CU_SOURCES LINK_LIBS)
  cmake_parse_arguments(PARSE_ARGV 0 TGT "${OPTS}" "${NVPO}" "${MVO}")

  if (TGT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments detected ${TGT_UNPARSED_ARGUMENTS}")
  endif()

  message(STATUS " + add_target ")
  message(STATUS " ++++++++++++++++++++++++++++++++++++++++++++++ ")

  if (TGT_CU_SOURCES)
    message(STATUS " + Compiling ${TGT_CU_SOURCES} with the CUDA compiler")
    set_source_files_properties(${TGT_CU_SOURCES} PROPERTIES LANGUAGE CUDA)
  endif()

  if (TGT_CXX_SOURCES)
    message(STATUS " + Compiling ${TGT_CXX_SOURCES} with the C++ compiler")
  endif()

  if (TGT_SHARED_LIB)
    add_library(${TGT_NAME} SHARED ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS " + Created shared library ${TGT_NAME}")
  elseif (TGT_STATIC_LIB)
    add_library(${TGT_NAME} STATIC ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS " + Created static library ${TGT_NAME}")
  elseif (TGT_LIB)
    add_library(${TGT_NAME} ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS " + Created library ${TGT_NAME}")
  elseif (TGT_EXEC)
    add_executable(${TGT_NAME} ${TGT_CU_SOURCES} ${TGT_CXX_SOURCES})
    message(STATUS " + Created executable ${TGT_NAME}")
  elseif (NOT NAME ${TGT_NAME})
    message(FATAL_ERROR "Target ${TGT_NAME} does not exist and was not created (missing LIB or EXEC).")
  endif()

  target_compile_features(${TGT_NAME} PUBLIC cxx_std_17)
  set_target_properties(${TGT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if (TGT_CUDA OR TGT_CU_SOURCES)
    if (NOT TGT_CU_ARCH)
      set(TGT_CU_ARCH "60;70;75")
    endif()
    set_target_properties(${TGT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${TGT_NAME} PROPERTIES CUDA_ARCHITECTURES "${TGT_CU_ARCH}")
    #target_link_options(${TGT_NAME} PRIVATE $<DEVICE_LINK:-fvisibility=hidden>) # <----- invalid device function without this

    message(STATUS " + Configured ${TGT_NAME} for CUDA ${TGT_CU_ARCH}")
 endif()

  if (TGT_LINK_LIBS)
    target_link_libraries(${TGT_NAME} PUBLIC ${TGT_LINK_LIBS})
    message(STATUS " + ${TGT_NAME} links to ${TGT_LINK_LIBS}")
  endif()

  message(STATUS " + ")
endfunction()





#
# this emulates the CUDA capable library we use as a git submodule
#
add_subdirectory(submod)

#
# this emulates our library
#
add_subdirectory(lib)

#
# this emulates our executable
#
add_subdirectory(test)

