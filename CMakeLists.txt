cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET luisa::compute)
    # download LuisaCompute from github.com/LuisaGroup/LuisaCompute.git to the build directory
    include(FetchContent)
    FetchContent_Populate(luisa_compute
            GIT_REPOSITORY "https://github.com/LuisaGroup/LuisaCompute.git"
            GIT_TAG next)
    set(LuisaCompute_SOURCE_DIR ${luisa_compute_SOURCE_DIR})
    message(STATUS "LuisaCompute source dir: ${LuisaCompute_SOURCE_DIR}")

    include(${LuisaCompute_SOURCE_DIR}/scripts/setup_output_dirs.cmake)

    project(GaussianSplatting)

    # add LuisaCompute as a subdirectory
    set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "Disable LuisaCompute tests")
    set(LUISA_COMPUTE_DOWNLOAD_NVCOMP OFF CACHE BOOL "Disable nvcomp download")
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "Use system STL")
    set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "Disable Rust bindings")
    set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "Disable CPU backend")
    set(LUISA_COMPUTE_ENABLE_REMOTE OFF CACHE BOOL "Disable remote backend")
    add_subdirectory(${LuisaCompute_SOURCE_DIR})
endif ()

# lib
file(GLOB_RECURSE LUISA_GAUSSIAN_SPLATTING_LIB_SOURCES CONFIGURE_DEPENDS lcgs/src/*.cpp)
file(GLOB_RECURSE LUISA_GAUSSIAN_SPLATTING_LIB_HEADERS CONFIGURE_DEPENDS lcgs/include/*.h)
add_library(luisa-gaussian-splatting-lib SHARED ${LUISA_GAUSSIAN_SPLATTING_LIB_SOURCES})
set_target_properties(luisa-gaussian-splatting-lib PROPERTIES PUBLIC_HEADER "${LUISA_GAUSSIAN_SPLATTING_LIB_HEADERS}")
target_compile_definitions(luisa-gaussian-splatting-lib PRIVATE LCGS_DLL_EXPORTS=1)
target_include_directories(luisa-gaussian-splatting-lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lcgs/include>)
target_link_libraries(luisa-gaussian-splatting-lib PUBLIC luisa::compute)

# test
file(GLOB_RECURSE LUISA_GAUSSIAN_SPLATTING_TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp)
add_executable(luisa-gaussian-splatting-test ${LUISA_GAUSSIAN_SPLATTING_TEST_SOURCES})
target_include_directories(luisa-gaussian-splatting-test PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/_framework>)
target_link_libraries(luisa-gaussian-splatting-test PRIVATE luisa::compute luisa-gaussian-splatting-lib)

# app
file(GLOB_RECURSE LUISA_GAUSSIAN_SPLATTING_APP_SOURCES CONFIGURE_DEPENDS app/*.cpp)
add_executable(luisa-gaussian-splatting ${LUISA_GAUSSIAN_SPLATTING_APP_SOURCES})
target_link_libraries(luisa-gaussian-splatting PRIVATE luisa::compute luisa-gaussian-splatting-lib)
