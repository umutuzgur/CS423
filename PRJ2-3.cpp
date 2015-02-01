

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <iostream>

using namespace cv;
using namespace std;
vector<pair<string, vector<Mat> > >* getHistogramFromDirectories();
Mat getHistogram(Mat *image);
float getHistIntersectionValue(Mat current, Mat compare);
float getChiSquareValue(Mat current, Mat compare);
#define QUANTUM 1
//QUANTUM value is controlled from here
#define GRID 1
//GRID 1 is just 1 cell and 4 is a 2x2 grid
#define COMPARE_METHOD 1
//CMPARE_METHOD 1 is hist intersection and 2 is chi square
int main(){
    
    vector<pair<string, vector<Mat > > >* histograms = getHistogramFromDirectories();
    vector<pair<string, vector<Mat > > >* test =  new vector<pair<string, vector<Mat > > >();
    vector<pair<string, vector<Mat > > >* trainning =  new vector<pair<string, vector<Mat > > >();
    
    vector<pair<string,int> > *results = new vector<pair<string, int> >();
    //Dividing histograms into 2 different containiers as test and trainning
    for (int i= 0 ; i<histograms->size(); i++) {
        test->push_back(pair<string, vector<Mat> >(histograms->at(i).first,vector<Mat>()));
        trainning->push_back(pair<string, vector<Mat> >(histograms->at(i).first,vector<Mat>()));
        
        for (int j = 0; j < histograms->at(i).second.size(); j++) {
            if(j < histograms->at(i).second.size()/2){
                trainning->at(i).second.push_back(histograms->at(i).second.at(j));
            }else
                test->at(i).second.push_back(histograms->at(i).second.at(j));
        }
        
    }
    //Finding the categories of histograms
    int indexRow = 0;
    for (vector<pair<string, vector<Mat > > >::iterator i = test->begin(); i != test->end(); i++) {
        results->push_back(pair<string, int>(i->first,0));
        int indexCol = 0;
        for (vector<Mat>::iterator j = i->second.begin(); j != i->second.end(); j++) {
            string category;
            float min = 100000000000000;
            //finding the closest min value to 0
            for (vector<pair<string, vector<Mat > > >::iterator t = trainning->begin(); t!=trainning->end(); t++) {
                float temp=0;
                for (vector<Mat>::iterator k = t->second.begin(); k != t->second.end(); k++) {
                    
                    if (COMPARE_METHOD==1) {
                        temp = getHistIntersectionValue(*j,*k);
                        
                    }else if(COMPARE_METHOD==2){
                        temp = getChiSquareValue(*j,*k);
                    }
                    if(temp<min){
                        min = temp;
                        category = t->first;
                    }
                    
                }
            }
            if(category==i->first)
            {
                results->at(indexRow).second += 1;
                //In case we have a hit in the grid formation. We skip to the next image
                if(GRID ==4){
                    int temp =indexCol%4;
                    j +=3-temp;
                    indexCol+= 3 - temp;
                    if (indexCol >= i->second.size()) {
                        break;
                    }
                }
            }
            indexCol++;
            
        }
        indexRow++;
    }
    
    //Printing the results
    int totalSize = 0;
    int totalCorrect = 0;
    for (int i = 0; i < results->size(); i++) {
        totalCorrect += results->at(i).second;
        float size;
        if(GRID==1){
            size = (double)results->at(i).second/test->at(i).second.size();
            totalSize += test->at(i).second.size();
        }else if(GRID==4){
            size = (double)results->at(i).second/(test->at(i).second.size()/4);
            totalSize += test->at(i).second.size()/4;
            
        }
        cout << results->at(i).first << "  " << size*100.0 << "%"<< endl;
    }
    cout << "Overall "<< ((float)totalCorrect/totalSize)*100.0 <<"%";
    
    
    return 0;
}


vector<pair<string, vector<Mat> > > *getHistogramFromDirectories(){
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
        Mat histogram;
        while (entry != NULL)
        {
            if (entry->d_type != DT_DIR && entry->d_name[0] != '.'){
                picture = imread(secondPath+entry->d_name,CV_LOAD_IMAGE_COLOR);
                if(GRID ==1){
                    //Getting the histogram of the image
                    histogram = getHistogram(&picture);
                    categorieHistograms->push_back(histogram);
                }else if(GRID==4){
                    //Dividing the image into 4 and getting the histograms of it
                    Mat leftUp = picture( Rect(0, 0, picture.cols/2, picture.rows/2) );
                    categorieHistograms->push_back(getHistogram(&leftUp));
                    Mat rightUp = picture( Rect(picture.cols/2, 0, picture.cols/2, picture.rows/2) );
                    categorieHistograms->push_back(getHistogram(&rightUp));
                    Mat leftDown = picture( Rect(0, picture.rows/2, picture.cols/2, picture.rows/2) );
                    categorieHistograms->push_back(getHistogram(&leftDown));
                    Mat rightDown = picture( Rect(picture.cols/2, picture.rows/2, picture.cols/2, picture.rows/2) );
                    categorieHistograms->push_back(getHistogram(&rightDown));
                }
            }
            entry = readdir(secondLevel);
        }
        histograms->push_back(pair<string, vector<Mat > >(categories->at(i), *categorieHistograms));
        closedir(secondLevel);
    }
    
    return histograms;
}
Mat getHistogram(Mat *image){
    int numPixels = image->rows*image->cols;
    int numberOfChannels = 3;
    Mat channels[3];
    Mat histogram = Mat::zeros(1, numberOfChannels*(256/QUANTUM), CV_32FC1);
    split(*image, channels);
    for (int i = 0; i < numberOfChannels; i++) {
        for (int j = 0; j < channels[0].rows; j++) {
            for (int k = 0; k < channels[0].cols; k++) {
                //Finding the histogram bin and increasing it by 1
                int density = channels[i].at<uchar>(j,k);
                int location =(int)(i+1)*(density/(double)QUANTUM);
                histogram.at<float>(0,location) += 1;
            }
        }
        
        
    }
    //Normalizing histogram
    for (int i = 0; i<histogram.cols ; i++) {
        
        histogram.at<float>(0,i) /= numPixels*numberOfChannels;
        
    }
    
    return histogram;
}
float getHistIntersectionValue(Mat current, Mat compare){
    //Applying histogram intersection formula
    float temp = 1;
    for (int t = 0 ; t<current.cols; t++) {
        temp -= MIN(current.at<float>(0,t),compare.at<float>(0,t));
    }
    return temp;
}

float getChiSquareValue(Mat current, Mat compare){
    //Applying chi square formula
    float temp = 0;
    for (int t = 0 ; t<current.cols; t++) {
        float x = current.at<float>(0,t);
        float y = compare.at<float>(0,t);
        float numerator = pow(x - y, 2.0);
        float denominator = x+ y;
        if (denominator==0) {
            continue;
        }
        temp += numerator/(denominator*2);
        
    }
    
    
    return temp;
}

