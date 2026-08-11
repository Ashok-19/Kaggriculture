cmake_minimum_required(VERSION 3.16)
project(kaggriculture_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(kag_agent SHARED
    src/market.cpp
    src/router.cpp
    src/controller.cpp
    src/c_api.cpp
)
target_include_directories(kag_agent PUBLIC include)
target_compile_options(kag_agent PRIVATE -O3 -Wall -Wextra -Wpedantic)
set_target_properties(kag_agent PROPERTIES OUTPUT_NAME kag_agent)

add_executable(kag_market_test tests/test_market.cpp src/market.cpp)
target_include_directories(kag_market_test PRIVATE include)
target_compile_options(kag_market_test PRIVATE -O2 -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME market_parity COMMAND kag_market_test)
