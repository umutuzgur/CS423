
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>
#include <dirent.h>

using namespace cv;
using namespace std;

Mat getDiceHistogram(Mat image);
double findRadius(vector<Point> contour , Point p1);
bool checkIfNew(vector<pair<Point, int> >  centers, Point p1);
bool containsBranch(vector< pair <Point,vector<Point> > > *pointMap, Point p);
bool contains1D(vector<Point> *points, Point p);
int findIndex(vector< pair <Point,vector<Point> > > pointMap, Point p);
bool contains2D(vector<vector<Point > > *points, Point p);
void findClusters(vector< pair <Point,vector<Point> > > *pointMap,  vector<Point> *points , vector<Point> *visitedPoints);

int main(){
    
    const char* PATH = "test";
    DIR *firstLevel = opendir(PATH);
    
    struct dirent *entry = readdir(firstLevel);
    Mat result;
    while (entry != NULL)
    {
        if (entry->d_type != DT_DIR && entry->d_name[0] != '.'){
            string path= "";
            path.append(PATH);
            path += "/";
            path += entry->d_name;
            Mat src = imread(path);
            try
            {
                result = getDiceHistogram(src);
                cout << entry->d_name << " : " << result << endl;
            }
            catch (exception& e)
            {
                cout << entry->d_name << " : " << "Error" << endl;
            }
            src.~Mat();
            
        }
        
        entry = readdir(firstLevel);
    }
    closedir(firstLevel);
    
    return 0;
    
}
Mat getDiceHistogram(Mat image){
    
    GaussianBlur(image, image, Size(5,5), 0, 0, BORDER_DEFAULT );
    Mat image_gray;
    Mat grad;
    /// Convert it to gray
    cvtColor( image, image_gray, CV_RGB2GRAY );
    
    threshold(image_gray, image_gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // Use canny to visualize the lines
    Canny(image_gray, grad, 20, 40,3);
    
    Mat temp = grad.clone();
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point> approx;
    
    
    /// Detect edges using canny
    /// Find contours
    findContours( temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    
    
    //Get Contours and Circles
    vector<pair<Point, int > > *centers = new vector<pair<Point, int > >();
    
    for( int i = 0; i< contours.size(); i++ )
    {
        approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) *0.02, true);
        //Check the area of it and corners found
        if(fabs(contourArea(contours[i])) < 70 || !isContourConvex(approx))
            continue;
        //More than 8 vertices means its likely a circle
        if (approx.size() >= 8)
        {
            Moments m = moments(contours[i], false);
            Point p1(m.m10/m.m00, m.m01/m.m00);
            //Get radius
            int radius = findRadius(contours[i],p1);
            //If newly found center is distant from the rest of the center
            if(checkIfNew(*centers, p1))
                centers->push_back(pair<Point, int> (p1,radius+3));
            
        }
    }
    
    //Get connected points on the same dices
    vector<pair<Point,Point> > *lines = new vector<pair<Point,Point> >();
    
    for (auto i = centers->begin(); i != centers->end(); i++) {
        
        for (auto j = centers->begin(); j != centers->end(); j++) {
            
            LineIterator it(image, i->first, j->first);
            bool accept = true;
            Vec3b *check = NULL;
            for(int t = 0; t < it.count; t++, ++it)
            {
                double distToCenter1 =norm(i->first - it.pos());
                double distToCenter2 =norm(j->first - it.pos());
                //If within the radius of these center continue because they are either black or white
                if(distToCenter1 <= i->second | distToCenter2 <= j->second  )
                    continue;
                if (check != NULL) {
                    Vec3b color2 = image.at<Vec3b>(it.pos());
                    int diff1 = abs((*check)[0] - color2[0]);
                    int diff2 = abs((*check)[1] - color2[1]);
                    int diff3 = abs((*check)[2] - color2[2]);
                    //Check difference
                    if(!(diff1<15 && diff2<15 && diff3<15)){
                        accept = false;
                        break;
                    }
                }
                //Set checking pixel
                if (t%5==0) {
                    check = &image.at<Vec3b>(it.pos());
                }
                
                
            }
            //If proper line accept it and save
            if(accept)
                lines->push_back(pair<Point,Point>(i->first,j->first));
            
        }
    }
    //Create point map
    vector< pair <Point,vector<Point> > > *pointMap = new  vector<pair<Point,vector<Point> > >();
    for (int i = 0; i < lines->size(); i++) {
        
        pair<Point,Point> line = lines->at(i);
        Point p1 = line.first;
        Point p2 = line.second;
        //Check if point the map if it is put p2 in that branch
        if (!containsBranch(pointMap, p1))
        {
            pair<Point, vector<Point> > point(p1, *new vector<Point >());
            point.second.push_back(p2);
            pointMap->push_back(pair<Point, vector<Point> >(p1, *new vector<Point >()) );
        }
        else
        {
            int location = findIndex(*pointMap, p1);
            vector<Point> points = pointMap->at(location).second;
            //Check if not present in that branch
            if(!contains1D(&points, p2))
                pointMap->at(location).second.push_back(p2);
        }
        // Do it for the reverse
        if (!containsBranch(pointMap, p2)) {
            pair<Point, vector<Point> > point(p2, *new vector<Point >());
            point.second.push_back(p1);
            pointMap->push_back(pair<Point, vector<Point> >(point) );
        }
        else
        {
            int location = findIndex(*pointMap, p2);
            vector<Point> points = pointMap->at(location).second;
            //Check if not present in that branch
            if(!contains1D(&points, p1))
                pointMap->at(location).second.push_back(p1);
            
        }
        
    }
    
    //Cluster the points
    vector< vector<Point> > *clusteredPoints = new  vector<vector<Point> >();
    
    for (int i = 0; i < pointMap->size(); i++) {
        vector<Point> *points = new vector<Point>();
        vector<Point> *visitedPoints = new vector<Point>();
        Point p = pointMap->at(i).first;
        
        //Check if point the clusters
        if(contains2D(clusteredPoints, p))
            continue;
        for (int j = 0;  j < pointMap->at(i).second.size(); j++) {
            points->push_back(pointMap->at(i).second.at(j));
        }
        //Assign visited
        visitedPoints->push_back(p);
        //Do it for the points
        findClusters(pointMap, points, visitedPoints);
        clusteredPoints->push_back(*points);
        
    }
    //Creates and fills the histogram
    Mat valueHistogram = Mat::zeros(1, 6, CV_8UC1);
    for (int i = 0; i <clusteredPoints->size() ; i++) {
        int size = clusteredPoints->at(i).size();
        valueHistogram.at<uchar>(0,size-1)++;
    }
    
    return valueHistogram;
}
void findClusters(vector< pair <Point,vector<Point> > > *pointMap,  vector<Point> *points , vector<Point> *visitedPoints){
    for (int i = 0; i< points->size(); i++) {
        Point p = points->at(i);
        if(!contains1D(visitedPoints, p)){
            
            visitedPoints->push_back(p);
            int index = findIndex(*pointMap, p);
            
            vector<Point> newPoints = pointMap->at(index).second;
            
            for (int j = 0; j < newPoints.size(); j++) {
                
                Point sub = newPoints.at(j);
                if(!contains1D(points, sub))
                    points->push_back(sub);
            }
            findClusters(pointMap, points, visitedPoints);
            
        }
        
    }
    
}

bool containsBranch(vector< pair <Point,vector<Point> > > *pointMap, Point p){
    bool contains = false;
    for(auto i = pointMap->begin() ; i != pointMap->end() ; i++){
        if(i->first == p){
            contains = true;
            break;
        }
        
    }
    return contains;
    
}
bool contains1D(vector<Point> *points, Point p){
    bool contains = false;
    for(auto i = points->begin() ; i != points->end() ; i++){
        if((*i)==p){
            contains = true;
            break;
        }
        
    }
    return contains;
    
}
bool contains2D(vector<vector<Point > > *reducedMap, Point p){
    bool contains = false;
    for(int i = 0 ; i < reducedMap->size() ; i++){
        for (int j = 0; j < reducedMap->at(i).size(); j++) {
            if(reducedMap->at(i).at(j)==p){
                contains = true;
                i = reducedMap->size();
                break;
            }
        }
        
        
    }
    return contains;
    
}



int findIndex(vector< pair <Point,vector<Point> > > pointMap, Point p){
    int index = 0;
    for(int i = 0 ; i < pointMap.size() ; i++){
        if(pointMap.at(i).first == p){
            index = i;
            break;
        }
        
    }
    return index;
    
}


double findRadius(vector<Point> contour , Point p1){
    double radius = 0;
    for (int i = 0; i < contour.size(); i++) {
        Point p2 = contour[i];
        double temp = norm(p1-p2);
        if(temp>radius)
            radius = temp;
    }
    return radius;
}
bool checkIfNew(vector<pair<Point, int> >  centers, Point p1){
    bool newCenter = true;
    for (auto i = centers.begin(); i != centers.end(); i++) {
        Point p2 = i->first;
        double diff = norm(p1-p2);
        if(diff<5){
            newCenter = false;
            break;
        }
        
    }
    return newCenter;
    
}

