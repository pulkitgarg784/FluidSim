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

# Dear ImGui v1.91.9b, vendored at external/imgui.
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/particle.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fluid_depth.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fluid_depth.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fluid_thickness.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fullscreen.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene_background.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene_mesh.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scene_mesh.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fluid_blur.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fluid_thickness_blur.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fluid_composite.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_external.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_reorder.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_start_reset.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_start_indices.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_density.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_pressure.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/sph_viscosity_integrate.comp)

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

add_executable(CS488
    src/main.cpp
    src/obj_loader.cpp
    src/vk_radix_sort_impl.cpp)
add_dependencies(CS488 shaders)

target_include_directories(CS488 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/vulkan_radix_sort/include
    ${Vulkan_INCLUDE_DIRS})

target_compile_definitions(CS488 PRIVATE
    SHADER_DIR="${SHADER_OUT_DIR}"
    ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(CS488 PRIVATE Vulkan::Vulkan imgui)
