#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <iostream>

using namespace cv;
using namespace std;

vector<pair<string, vector<Mat> > >* getHistogramFromDirectoriesForMean();
Mat getHistogramForMean(Mat *image);
float getGaus(double dist, float sigma);

#define QUANTUM 8

int main(){
    vector<pair<string, vector<Mat > > >* histograms = getHistogramFromDirectoriesForMean();
    float sigma = 0.03;
    float threshold = 0.00007;
    Mat *dataForPCA = new Mat();
    
    for (int i= 0 ; i<histograms->size(); i++) {
        for (int j = 0; j < histograms->at(i).second.size(); j++) {
            dataForPCA->push_back(histograms->at(i).second.at(j));
        }
    }
    PCA pca(*dataForPCA, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.85);
    
    for (int i = 0; i < dataForPCA->rows; i++) {
        Mat point = pca.project(dataForPCA->row(i));
        dataForPCA->row(i) = point;
    }
    Mat means;
    Mat average;
    Mat current = dataForPCA->row(0);
    int i = 1;
    while (i < dataForPCA->rows) {
        average = current.clone();
        
        float totalWeight= 0;
        
        for (int y = 0; y < dataForPCA->rows; y++) {
            if(i==y)
                continue;
            Mat second = dataForPCA->row(y);
            float dist = norm(second, current, NORM_L2);
            
            float weight = getGaus(dist, sigma);
            totalWeight += weight;
            average += second * weight;
            
            
        }
        average = average/totalWeight;
        double dist = norm(average, current, NORM_L2);
        if(dist < threshold){
            bool maxima = true;
            for (int t = 0; t < means.rows; ++t) {
                Mat currentMean = means.row(t);
                double distBetweenMeans = norm(average ,currentMean, NORM_L2);
                if (distBetweenMeans < 0.00001) {
                    currentMean = (currentMean + average) / 2.0;
                    maxima = false;
                    break;
                }
            }
            if(maxima)
                means.push_back(average);
            current = dataForPCA->row(i++);
        }else
            current = average;
        
        
    }
    
    cout << "Number of center:" <<means.rows;
    
    
    return 0;
}



float getGaus(double dist, float sigma){
    float numerator = exp(-1.0 * pow(dist,2) / (2.0 * pow(sigma, 2)));
    float denominator = (sigma * sqrt(2.0 * CV_PI));
    
    return numerator/denominator;
}



vector<pair<string, vector<Mat> > > *getHistogramFromDirectoriesForMean(){
    vector<pair<string, vector<Mat > > >* histograms = new vector<pair<string, vector< Mat > > >();
    const char* PATH = "DATASET";
    vector<string>* categories = new vector<string>();
    DIR *firstLevel = opendir(PATH);
    
    struct dirent *entry = readdir(firstLevel);
    
    while (entry != NULL)
    {
        if (entry->d_type == DT_DIR && entry->d_name[0] != '.')
            categories->push_back(entry->d_name);
        
        entry = readdir(firstLevel);
    }
    closedir(firstLevel);
    
    for (int i = 0; i < categories->size(); i++) {
        string secondPath = "";
        secondPath+=PATH;
        secondPath+= "/" + categories->at(i) + "/";
        
        
        DIR *secondLevel = opendir(secondPath.c_str());
        entry = readdir(secondLevel);
        vector<Mat> *categorieHistograms = new vector<Mat>();
        Mat picture;
        while (entry != NULL)
        {
            if (entry->d_type != DT_DIR && entry->d_name[0] != '.'){
                picture = imread(secondPath+entry->d_name,CV_LOAD_IMAGE_COLOR);
                Mat *wholeImage = new Mat();
                for (int x = 0; x <= picture.cols- picture.cols/4 ;x +=picture.cols/4) {
                    for (int y = 0; y <= picture.rows - picture.rows/4; y +=picture.rows/4) {
                        
                        Mat image = picture( Rect(x, y, picture.cols/4, picture.rows/4) );
                        if(wholeImage->empty())
                            *wholeImage = getHistogramForMean(&image);
                        else
                            hconcat(*wholeImage, getHistogramForMean(&image), *wholeImage);
                        
                    }
                    
                }
                categorieHistograms->push_back(*wholeImage);
                
                
            }
            entry = readdir(secondLevel);
        }
        histograms->push_back(pair<string, vector<Mat > >(categories->at(i), *categorieHistograms));
        closedir(secondLevel);
    }
    
    return histograms;
}
Mat getHistogramForMean(Mat *image){
    int numPixels = image->rows*image->cols;
    int numberOfChannels = 3;
    Mat histogram = Mat::zeros(1, numberOfChannels*(256/QUANTUM), CV_32FC1);
    for(int row = 0; row < image->rows; ++row) {
        uchar* p = image->ptr(row);
        
        for(int col = 0; col < image->cols*3; ++col) {
            //Finding the histogram bin and increasing it by 1
            int density = *p++;
            int location =(int)(col/image->cols+1)*(density/(double)QUANTUM);
            float* pHistogram = histogram.ptr<float>(0,location);
            *pHistogram += 1;
            
        }
        
    }
    //Normalizing histogram
    float* p = histogram.ptr<float>(0);
    for(int col = 0; col < histogram.cols; ++col) {
        //Finding the histogram bin and increasing it by 1
        *p /= numPixels*numberOfChannels*16;
        *p++;
        
    }
    
    
    
    return histogram;
}
