cmake_minimum_required(VERSION 3.0.0)
if (WIN32)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

endif (WIN32)
if (UNIX)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)
endif (UNIX)

project(brute_force_algorithm LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
	${PROJECT_SOURCE_DIR}/include
)
enable_testing()

file(GLOB all_SRCS
	"${PROJECT_SOURCE_DIR}/src/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/*.c"
	"${PROJECT_SOURCE_DIR}/src/*.cu"
)

# Agregue un origen al ejecutable de este proyecto.
add_executable (brute_force_algorithm ${all_SRCS})

# Copiar carpeta data
add_custom_command(TARGET brute_force_algorithm PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:brute_force_algorithm>/data)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
FetchContent_MakeAvailable(json)

target_link_libraries(brute_force_algorithm PRIVATE nlohmann_json::nlohmann_json)

set_target_properties(brute_force_algorithm PROPERTIES LINKER_LANGUAGE CXX)

set_property(TARGET brute_force_algorithm PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

target_compile_options(brute_force_algorithm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo --default-stream per-thread 
            --expt-relaxed-constexpr
            -Xptxas 
            -dlcm=cg
            -ftz=true 
            -prec-div=false 
            -prec-sqrt=false>) 

# Gprof 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
include(CPack)

