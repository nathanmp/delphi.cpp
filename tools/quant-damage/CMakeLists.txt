# llama-quant-damage-impl: sampled quantization damage estimator, reusable by tests

set(TARGET llama-quant-damage-impl)

add_library(${TARGET} STATIC quant-damage.cpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

# llama-quant-damage executable

set(TARGET llama-quant-damage)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-quant-damage-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# llama-quant-damage-kld: single-tensor KLD, to calibrate the estimate against model-level damage

set(TARGET llama-quant-damage-kld)

add_executable(${TARGET} kld.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE llama-quant-damage-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
