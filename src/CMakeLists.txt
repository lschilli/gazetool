#SET(MOC_HEADERS
#    glimageview.h
#    gazergui.h
#)

QT5_WRAP_UI(UI_HEADERS gazergui.ui)

SET(DLIBGAZER_SRC
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
    SET(DLIBGAZER_SRC ${DLIBGAZER_SRC} yarpsupport.cpp)
ENDIF()

ADD_EXECUTABLE(dlibgazer ${DLIBGAZER_SRC})
ADD_DEPENDENCIES(dlibgazer ${UI_HEADERS})
TARGET_LINK_LIBRARIES(dlibgazer dlib GL ${Boost_LIBRARIES} ${YARP_LIBRARIES} ${OpenCV_LIBS})
qt5_use_modules(dlibgazer Core Widgets Gui OpenGL)

INSTALL(TARGETS dlibgazer
  RUNTIME DESTINATION bin
)
