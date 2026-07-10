cmake_minimum_required(VERSION 3.16)
project(CS488 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

# GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# Shader compilation
find_program(GLSLC glslc HINTS
    $ENV{VULKAN_SDK}/bin
    /usr/local/bin
    /opt/homebrew/bin)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found (part of the Vulkan SDK)")
endif()

set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/particle.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/particle.frag)

set(SPIRV_OUTPUTS)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(FNAME ${SHADER} NAME)
    set(SPV ${SHADER_OUT_DIR}/${FNAME}.spv)
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader ${FNAME}")
    list(APPEND SPIRV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders DEPENDS ${SPIRV_OUTPUTS})

add_executable(CS488 src/main.cpp)
add_dependencies(CS488 shaders)

target_include_directories(CS488 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS})

target_compile_definitions(CS488 PRIVATE
    SHADER_DIR="${SHADER_OUT_DIR}")

target_link_libraries(CS488 PRIVATE Vulkan::Vulkan glfw)
