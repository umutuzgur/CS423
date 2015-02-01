
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
#define QUANTUM 8
//QUANTUM value is controlled from here

int main(){
    
    vector<pair<string, vector<Mat > > >* histograms = getHistogramFromDirectories();
    vector<pair<string, vector<Mat > > >* test =  new vector<pair<string, vector<Mat > > >();
    vector<pair<string, vector<Mat > > >* trainning =  new vector<pair<string, vector<Mat > > >();
    
    Mat  *trainningForPCA =  new Mat();
    
    vector<pair<string,int> > *results = new vector<pair<string, int> >();
    //Dividing histograms into 2 different containiers as test and trainning
    
    for (int i= 0 ; i<histograms->size(); i++) {
        test->push_back(pair<string, vector<Mat> >(histograms->at(i).first,vector<Mat>()));
        trainning->push_back(pair<string, vector<Mat> >(histograms->at(i).first,vector<Mat>()));
        for (int j = 0; j < histograms->at(i).second.size(); j++) {
            if(j < histograms->at(i).second.size()/2){
                trainningForPCA->push_back(histograms->at(i).second.at(j));
                trainning->at(i).second.push_back(histograms->at(i).second.at(j));
            }else{
                test->at(i).second.push_back(histograms->at(i).second.at(j));
            }
        }
        
    }
    
    
    PCA pca(*trainningForPCA, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.8);
    for (vector<pair<string, vector<Mat > > >::iterator i = test->begin(); i != test->end(); i++) {
        for (vector<Mat>::iterator j = i->second.begin(); j != i->second.end(); j++) {
            *j = pca.project(*j);
        }
    }
    for (vector<pair<string, vector<Mat > > >::iterator i = trainning->begin(); i != trainning->end(); i++) {
        for (vector<Mat>::iterator j = i->second.begin(); j != i->second.end(); j++) {
            *j = pca.project(*j);
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
                    
                    temp = norm(*k,*j,NORM_L2);
                    
                    
                    if(temp<min){
                        min = temp;
                        category = t->first;
                    }
                    
                }
            }
            if(category==i->first)
                results->at(indexRow).second += 1;
            
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
        size = (double)results->at(i).second/(test->at(i).second.size());
        totalSize += test->at(i).second.size();
        
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
                Mat *wholeImage = new Mat();
                for (int x = 0; x <= picture.cols- picture.cols/4 ;x +=picture.cols/4) {
                    for (int y = 0; y <= picture.rows - picture.rows/4; y +=picture.rows/4) {
                        
                        Mat image = picture( Rect(x, y, picture.cols/4, picture.rows/4) );
                        if(wholeImage->empty())
                            *wholeImage = getHistogram(&image);
                        else
                            hconcat(*wholeImage, getHistogram(&image), *wholeImage);
                        
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
Mat getHistogram(Mat *image){
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

