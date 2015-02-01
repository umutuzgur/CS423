#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/** @function main */
int main5( int argc, char** argv )
{
    
    Mat src, src_gray;
    Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    
    
    /// Load an image
    src = imread( argv[1] );
    
    if( !src.data )
    { return -1; }
    
    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    /// Convert it to gray
    cvtColor( src, src_gray, CV_RGB2GRAY );
    
    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    dilate(abs_grad_x, abs_grad_x, Mat());
    
    
    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    dilate(abs_grad_y, abs_grad_y, Mat());
    
    
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad );
    erode(grad, grad, Mat());
    cv::threshold(grad, grad, 150, 255, CV_THRESH_BINARY );
    
  
    vector<Vec4i> lines;
    HoughLinesP(grad, lines, 1, CV_PI/180, 50, 50, 10);
    cvtColor(grad, grad, CV_GRAY2BGR);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( grad, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
    
    imshow( "test", grad );
    
    waitKey(0);
    
    return 0;
}