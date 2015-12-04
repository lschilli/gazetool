SET(GAZE_LIB_SRC
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
    SET(GAZE_LIB_SRC ${GAZE_LIB_SRC} yarpsupport.cpp)
ENDIF()


ADD_LIBRARY(gaze SHARED ${GAZE_LIB_SRC})
target_link_libraries(gaze ${OpenCV_LIBS} ${dlib_LIBRARIES} ${Boost_LIBRARIES} ${YARP_LIBRARIES})

IF (ENABLE_QT5)
    qt5_use_modules(gaze Core)
ENDIF()


export(TARGETS gaze
  FILE "${PROJECT_BINARY_DIR}/gazeTargets.cmake")
export(PACKAGE gaze)


set(INSTALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/gaze/")

configure_file("${GAZE_CONFIG_DIR}/gazeConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazeConfig.cmake" @ONLY)
configure_file("${GAZE_CONFIG_DIR}/gazeConfigVersion.cmake.in"
  "${PROJECT_BINARY_DIR}/gazeConfigVersion.cmake" @ONLY)

export(TARGETS gaze
  FILE "${PROJECT_BINARY_DIR}/gazeTargets.cmake")

INSTALL(TARGETS gaze
     LIBRARY DESTINATION lib
)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "include/gaze"
    FILES_MATCHING
    PATTERN "*.h")

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazeConfig.cmake"
  "${PROJECT_BINARY_DIR}/gazeConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/gazeTargets.cmake"
  DESTINATION "lib/cmake/gaze" COMPONENT dev)
