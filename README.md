## CMake should pass `-fvisibility=hidden` during device linking.

this project reproduces a bug when linking shared and static libraries that use CUDA into an executable.

### libraries and executables:
```
submod : a static library containing a mix of C++ and CUDA code
core : a static library containing a mix of C++ and CUDA code
extension : a **shared** library containing only C++ code. this must be shared and compiled separately.
public: a static library that conatins the management code for calling using core and extension
exec : a simple command line executable program that calls functions in the library
```
### dependencies
```
core depends on: submod
extension depends on: core
public depends on: core and extension
exec depends on: public
```

If all the libraries are shared libraries there is not issue, but if the libraries are both shared and static libraries then the code errors or crashes at runtime.


explicitly adding `-fvisibility=hidden` during device linking fixes the issue. The bug is that CMake should pass this flag during device linking.
Setting `CMAKE_CUDA_VISIBILITY_PRESET hidden` should result in the `-fvisibility=hidden` option being passed in the device linking step.

in this reproducer the bug manifests itself as "invalid device function" when invoking CUDA kernels in the libraries.
the reproducer is structured to mimic a real world project that was experiencing a SEGV before main during CUDA library initialization.




To demonstrate the issue:
```
git clone git@github.com:burlen/device_link_issue.git
cd device_link_issue
mkdir bin
cd bin
rm -rfI ./*; cmake -DBUILD_SHARED_LIBS=OFF ..
make VERBOSE=1
./test/exec
```
this compiles such that all but one of the internal libraries are static and the one is shared.
the run will error out with "invlaid device function" error.

Explicitly adding `target_link_options(${TGT_NAME} PRIVATE $<DEVICE_LINK:-fvisibility=hidden>)` on line 83 of the CMakeLists.txt file resolves the issue and the code runs correctly without error.


## System Details:
```
cmake version 3.22.2

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_Mar__8_18:18:20_PST_2022
Cuda compilation tools, release 11.6, V11.6.124
Build cuda_11.6.r11.6/compiler.31057947_0

g++ (GCC) 11.2.1 20220127 (Red Hat 11.2.1-9)

03:00.0 VGA compatible controller: NVIDIA Corporation TU104GL [Quadro RTX 4000] (rev a1)

NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6
```

