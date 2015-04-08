cmake_minimum_required( VERSION 2.8 )

project( OnlineAnalysisWinding )
#set( PROJECT_VERSION 0.0.0.0 )
#set( PROJECT_BRIEF "\"No description\"" )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" )
set( LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib" )
set( EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" )

# ROOT, a library for HEP data analysis (contains RooFit)
find_package( ROOT REQUIRED )
include_directories( ${ROOT_INCLUDE_DIR} )
set( LIBRARIES ${LIBRARIES} ${ROOT_LIBRARIES} )
set( LIBRARIES ${LIBRARIES} ${ROOFIT_LIBRARIES} )

# Find and include OpenCV directories
find_package( OpenCV 2.4.10 REQUIRED )
if( ${OpenCV_FOUND} )

  MESSAGE( STATUS "OpenCV version: " ${OpenCV_VERSION} )

endif( ${OpenCV_FOUND} )

include_directories( ${OpenCV_INCLUDE_DIRS} )
set( LIBRARIES ${LIBRARIES} ${OpenCV_LIBS} )

# Find and include Boost Libraries
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
set( LIBRARIES ${LIBRARIES} ${Boost_LIBRARIES} )

# Include mvImpact acquire SDK
find_package( mvIMPACT REQUIRED )
INCLUDE_DIRECTORIES( ${mvIMPACT_INCLUDE_DIRS} )
set( LIBRARIES ${LIBRARIES} ${mvIMPACT_LIBRARIES} )

if( ${APPLE} )

  MESSAGE( STATUS "Building for Mac OS X, switching on C++11 flags for Mac OS X/clang" )
  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall -Wextra" )

endif( ${APPLE} )

if( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )

  MESSAGE( STATUS "Building for Linux, switching on C++11 flags for Linux/gcc" )
  SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -pedantic -Wextra" )

endif( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )


add_executable( OpenCVCam OpenCVCam.cpp )
target_link_libraries( OpenCVCam ${LIBRARIES} )
