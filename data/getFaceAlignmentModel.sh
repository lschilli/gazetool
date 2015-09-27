#!/bin/bash

if [ ! -f shape_predictor_68_face_landmarks.dat ]; then
	wget http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
	bunzip2 shape_predictor_68_face_landmarks.dat.bz2
fi
