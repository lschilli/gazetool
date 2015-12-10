cmake_minimum_required(VERSION 2.8)

PROJECT(libgazetool)

set(GAZETOOL_LIB_MAJOR_VERSION 0)
set(GAZETOOL_LIB_MINOR_VERSION 1)
set(GAZETOOL_LIB_PATCH_VERSION 0)
set(GAZETOOL_LIB_VERSION
  ${GAZETOOL_LIB_MAJOR_VERSION}.${GAZETOOL_LIB_MINOR_VERSION}.${GAZETOOL_LIB_PATCH_VERSION})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

OPTION(ENABLE_YARP_SUPPORT "Enable yarp support" OFF)
OPTION(ENABLE_QT5 "Build demo (QT5 is required)" ON)

set(CMAKE_AUTOMOC TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

FIND_PACKAGE(dlib REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system chrono program_options REQUIRED)

IF (ENABLE_QT5)
    FIND_PACKAGE(Qt5Core REQUIRED)
    FIND_PACKAGE(Qt5Widgets REQUIRED)
    FIND_PACKAGE(Qt5OpenGL REQUIRED)
    FIND_PACKAGE(Qt5Gui REQUIRED)
    ADD_DEFINITIONS(-DENABLE_QT5)
ENDIF()

IF(ENABLE_YARP_SUPPORT)
    FIND_PACKAGE(YARP REQUIRED)
ENDIF()

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${YARP_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})

SET(GAZETOOL_LIB_SRC
    imageprovider.cpp
    faceparts.cpp
    pupilfinder.cpp
    eyelidlearner.cpp
    mutualgazelearner.cpp
    relativeeyelidlearner.cpp
    relativegazelearner.cpp
    verticalgazelearner.cpp
    facedetectionworker.cpp
    shapedetectionworker.cpp
    gazehyps.cpp
    regressionworker.cpp
    eyepatcher.cpp
    featureextractor.cpp
    abstractlearner.cpp
    rlssmoother.cpp
    blockingqueue.h
)

IF(ENABLE_YARP_SUPPORT)
    ADD_DEFINITIONS(-DENABLE_YARP_SUPPORT)
    SET(GAZETOOL_LIB_SRC ${GAZETOOL_LIB_SRC} yarpsupport.cpp)
ENDIF()


ADD_LIBRARY(gazetool SHARED ${GAZETOOL_LIB_SRC})
target_link_libraries(gazetool ${OpenCV_LIBS} ${dlib_LIBRARIES} ${Boost_LIBRARIES} ${YARP_LIBRARIES})

IF (ENABLE_QT5)
    qt5_use_modules(gazetool Core)
ENDIF()

set(INSTALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/gazetool/")

configure_file("config/gazetoolConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazetoolConfig.cmake" @ONLY)
configure_file("config/gazetoolConfigVersion.cmake.in"
  "${PROJECT_BINARY_DIR}/gazetoolConfigVersion.cmake" @ONLY)

export(TARGETS gazetool
  FILE "${PROJECT_BINARY_DIR}/gazetoolTargets.cmake")
export(PACKAGE gazetool)

INSTALL(TARGETS gazetool
     LIBRARY DESTINATION lib
)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "include/gazetool"
    FILES_MATCHING
    PATTERN "*.h")

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazetoolConfig.cmake"
  "${PROJECT_BINARY_DIR}/gazetoolConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/gazetoolTargets.cmake"
  DESTINATION "lib/cmake/gazetool" COMPONENT dev)
