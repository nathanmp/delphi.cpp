# llama-perplexity-impl: perplexity logic, reusable by app

add_library(llama-perplexity-hotswap STATIC hotswap.cpp)
set_target_properties(llama-perplexity-hotswap PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(llama-perplexity-hotswap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(llama-perplexity-hotswap PUBLIC llama-common llama)
target_compile_features(llama-perplexity-hotswap PUBLIC cxx_std_17)

set(TARGET llama-perplexity-impl)

add_library(${TARGET} perplexity.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama llama-perplexity-hotswap ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-perplexity executable

set(TARGET llama-perplexity)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-perplexity-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
