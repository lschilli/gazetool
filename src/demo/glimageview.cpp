#include "glimageview.h"

GLImageView::GLImageView(QWidget *parent) :
        QGLWidget(parent), format(GL_BGR), depth(GL_UNSIGNED_BYTE) {
    bgColor = this->palette().color(QPalette::Window);
    initializeGL();
}

GLImageView::~GLImageView() {
	glDeleteTextures(1, &texture);
}


void GLImageView::initializeGL() {
    qglClearColor(bgColor);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, this->width(), this->height());
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, this->width(), this->height(), 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glEnable (GL_TEXTURE_2D);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glDisable(GL_TEXTURE_2D);
}

void GLImageView::paintGL() {
	GLfloat offset_x = 0;
	GLfloat offset_y = 0;
	GLfloat gl_width = this->width();
	GLfloat gl_height = this->height();

	float ratioImg = float(cv_frame.cols) / (float) cv_frame.rows;
	float ratioScreen = this->width() / (float) this->height();

	if (ratioImg > ratioScreen)
		gl_height = gl_width / ratioImg;
	else
		gl_width = gl_height * ratioImg;
    offset_x = (this->width() - gl_width) * .5f;
    //offset_y = (this->height() - gl_height) * .5f;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable (GL_DEPTH_TEST);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, this->width(), this->height(), 0.0f, 0.0f, 1.0f);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity();
	glEnable (GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cv_frame.cols, cv_frame.rows, 0, format, depth, cv_frame.ptr());
	glBegin (GL_QUADS);
	glTexCoord2i(0, 1);
	glVertex2i(offset_x, gl_height + offset_y);
	glTexCoord2i(0, 0);
	glVertex2i(offset_x, offset_y);
	glTexCoord2i(1, 0);
	glVertex2i(gl_width + offset_x, offset_y);
	glTexCoord2i(1, 1);
	glVertex2i(gl_width + offset_x, gl_height + offset_y);
	glEnd();
}

void GLImageView::resizeGL(int width, int height) {
	glViewport(0, 0, this->width(), this->height());
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, this->width(), this->height(), 0.0f, 0.0f, 1.0f);
	glMatrixMode (GL_MODELVIEW);
    glLoadIdentity();
}

QSize GLImageView::sizeHint() const
{
    return QSize(cv_frame.cols, cv_frame.rows);
}

void GLImageView::setImage(const cv::Mat& frame) {
    switch (frame.channels()) {
	case 1:
        format = GL_LUMINANCE;
		break;
	case 2:
		format = GL_LUMINANCE_ALPHA;
		break;
	case 3:
		format = GL_BGR;
		break;
	case 4:
		format = GL_BGRA;
		break;
	default:
		return;
	}
    switch (frame.depth()) {
    case CV_8U:
        depth = GL_UNSIGNED_BYTE;
        break;
    case CV_16U:
        depth = GL_UNSIGNED_SHORT;
        break;
    default:
        return;
    }
    cv::Size oldsize = cv_frame.size();
    cv_frame = frame;
    if (cv_frame.size() != oldsize) updateGeometry();
    update();
}
