cmake_minimum_required(VERSION 2.8)

PROJECT(gazetool)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

OPTION(ENABLE_YARP_SUPPORT "Enable yarp support" OFF)
OPTION(ENABLE_INSTALL_DLIB_FACE_MODEL "Install the dlib face landmarks model" OFF)
OPTION(ENABLE_QT5 "Enable QT5 support" ON)

option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  OFF)
if(USE_AVX_INSTRUCTIONS)
     add_definitions(-mavx)
endif()

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

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${YARP_INCLUDE_DIRS} ${dlib_INCLUDE_DIRS})

ADD_SUBDIRECTORY(src/lib)
ADD_SUBDIRECTORY(src)


CONFIGURE_FILE(gazetool.sh.in gazetool.sh @ONLY)

INSTALL(FILES
    data/gaze_est_deg.dat
    data/lid_est.dat
    data/vertgaze_est_deg.dat
    DESTINATION share/gazetool)

IF(ENABLE_INSTALL_DLIB_FACE_MODEL)
    INSTALL(FILES
        data/shape_predictor_68_face_landmarks.dat
        DESTINATION share/gazetool)
ENDIF()

INSTALL(PROGRAMS ${CMAKE_BINARY_DIR}/gazetool.sh DESTINATION bin)
