/*Project 1 is developed for applying linear filter to images. 
It is run by the command line by adding the path of the image and the csv file.
The format of the csv file should be like below
1,2,3
4,5,6
7,8,9
*/
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>


using namespace std;
using namespace cv;
Mat applyFilter(Mat image, Mat filter);
int N;

int main(int argc, char** argv )
{
    Mat src;
    Mat dest;
    src = imread( argv[1], 1 );
    //We read the csv file
    CvMLData mlData;
    mlData.read_csv(argv[2]);
    const CvMat* tmp = mlData.get_values();
    Mat filterMatrix(tmp, true);
    filterMatrix.convertTo(filterMatrix, CV_32FC1);
    N = filterMatrix.cols/2;
    
    if ( !src.data )
    {
        printf("No image data \n");
        return -1;
    }
    dest = applyFilter(src,filterMatrix);
    
    namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
    imshow("Display Image", dest);
    imwrite(argv[3], dest);
    waitKey(0);
    
    
    return 0;
    
}
Mat applyFilter(Mat image, Mat filter){
    
    //Apply Filter to the image
    Mat filterApplied(image.rows, image.cols, CV_32FC3, Scalar(0,0,0));
    for(int x = N ; x< image.cols-N;x++){
        for (int y = N; y < image.rows-N; y++)
        {
            float blue = 0;
            float green = 0;
            float red = 0;
            
            for (int filterx = -N; filterx < filter.cols-N; filterx++)
            {
                for (int filtery = -N; filtery < filter.rows-N; filtery++)
                {
                    Vec3b imagePixel = image.at<Vec3b>(y+filtery,x+filterx);
                    float filterConstant = filter.at<float>(filtery+N,filterx+N);
                    blue += imagePixel[0]*filterConstant;
                    green += imagePixel[1]*filterConstant;
                    red += imagePixel[2]*filterConstant;
                    
                }
            }
            Vec3f newPixel(blue,green,red);
            
            filterApplied.at<Vec3f>(y,x) = newPixel;
        }
    }
    // Find Max and Min Values
    float min = 1000000000;
    float max = -100000000;
    
    Mat cropedImage = filterApplied(Rect(N,N,filterApplied.rows-2*N,filterApplied.cols-2*N));
    for (int i = 0; i < cropedImage.rows; ++i)
    {
        for (int j = 0; j < cropedImage.cols; ++j)
        {
            if(cropedImage.at<Vec3f>(i,j)[0]>max)
                max = cropedImage.at<Vec3f>(i,j)[0];
            if(cropedImage.at<Vec3f>(i,j)[1]>max)
                max = cropedImage.at<Vec3f>(i,j)[1];
            if(cropedImage.at<Vec3f>(i,j)[2]>max)
                max = cropedImage.at<Vec3f>(i,j)[2];
            
            if (cropedImage.at<Vec3f>(i,j)[0]<min)
                min = cropedImage.at<Vec3f>(i,j)[0];
            if (cropedImage.at<Vec3f>(i,j)[1]<min)
                min = cropedImage.at<Vec3f>(i,j)[1];
            if (cropedImage.at<Vec3f>(i,j)[2]<min)
                min = cropedImage.at<Vec3f>(i,j)[2];
            
            
        }
    }
    
    //Change the ratio of the range
    Mat final = Mat::zeros( cropedImage.rows, cropedImage.cols, CV_8UC3 );
    float numerator = max-min;
    
    for (int i = 0; i< cropedImage.rows; i++) {
        for (int j = 0; j< cropedImage.cols; j++) {
            float blue  = cropedImage.at<Vec3f>(i,j)[0];
            float green  = cropedImage.at<Vec3f>(i,j)[1];
            float red  = cropedImage.at<Vec3f>(i,j)[2];
            final.at<Vec3b>(i,j)[0] = (((blue-min)*255.0f)/numerator);
            final.at<Vec3b>(i,j)[1] = (((green-min)*255.0f)/numerator);
            final.at<Vec3b>(i,j)[2] = (((red-min)*255.0f)/numerator);
                        
        }
        
    }
    return final;
    
}