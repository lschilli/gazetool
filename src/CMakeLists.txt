QT5_WRAP_UI(UI_HEADERS gazergui.ui)

SET(GAZETOOL_SRC
    main.cpp
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
    gazergui.cpp
    glimageview.cpp
    workerthread.cpp
    eyepatcher.cpp
    featureextractor.cpp
    abstractlearner.cpp
    rlssmoother.cpp
    ${UI_HEADERS}
    blockingqueue.h
)


IF(ENABLE_YARP_SUPPORT)
    ADD_DEFINITIONS(-DENABLE_YARP_SUPPORT)
    SET(GAZETOOL_SRC ${GAZETOOL_SRC} yarpsupport.cpp)
ENDIF()

ADD_EXECUTABLE(gazetool ${GAZETOOL_SRC})
ADD_DEPENDENCIES(gazetool ${UI_HEADERS})
TARGET_LINK_LIBRARIES(gazetool GL ${Boost_LIBRARIES} ${YARP_LIBRARIES} ${OpenCV_LIBS} ${dlib_LIBRARIES})
qt5_use_modules(gazetool Core Widgets Gui OpenGL)

INSTALL(TARGETS gazetool
  RUNTIME DESTINATION bin
)
