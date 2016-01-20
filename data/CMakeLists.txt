cmake_minimum_required(VERSION 2.8)

PROJECT(gazetool_data)

OPTION(ENABLE_INSTALL_DLIB_FACE_MODEL "Install the dlib face landmarks model" ON)


INSTALL(FILES
    gaze_est_deg.dat
    lid_est.dat
    vertgaze_est_deg.dat
    DESTINATION share/gazetool)


IF(ENABLE_INSTALL_DLIB_FACE_MODEL)
    INSTALL(FILES
       shape_predictor_68_face_landmarks.dat
       DESTINATION share/gazetool)
ENDIF()
