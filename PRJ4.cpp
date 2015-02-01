/*Project 4 is designed for analyzing the time from a clock.
 It works better with clocks that dont have a second hand because it works with k-means algortihm*/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <dirent.h>

using namespace cv;
using namespace std;


bool checkIfInTheRange(Vec4i l,int thresh,int centerX,int centerY);
Point2f getIntersectionOfLines(Vec4i line1, Vec4i line2);
Point getDominantCordinate(Vec4i line, Point p);
double getLength(Vec4i l);
void addPointTo(Vec4i l, Mat *matPoint);
int findMaxLengthIndex(vector<Vec4i> lines);
bool segmentIntersectRectangle(double a_p1x, double a_p1y, double a_p2x, double a_p2y, Mat src);
string getTimeFromClock(Mat src);

#define RAD 57.295





int main(int argc, char** argv)
{
    const char* PATH = "DATASET";
    DIR *firstLevel = opendir(PATH);
    
    struct dirent *entry = readdir(firstLevel);
    string result;
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
                result = getTimeFromClock(src);
            }
            catch (exception& e)
            {
                result = "Error";
            }
            src.~Mat();
            cout << entry->d_name << "  " << result << endl;
        }
        
        entry = readdir(firstLevel);
    }
    closedir(firstLevel);
    return 0;
}
string getTimeFromClock(Mat src){
    
    Mat src_gray;
    Mat matPoint;
    cvtColor(src, src_gray, CV_BGR2GRAY);
    //Eliminate minor details
    cv::threshold(src_gray, src_gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    
    //Invert colors
    bitwise_not ( src_gray, src_gray );
    
    
    
    //Try to eliminate second hand if present
    erode(src_gray, src_gray, Mat(),Point(-1,-1));

    
    
    vector<Vec4i> lines;
    vector<Vec4i> validLines;
    HoughLinesP(src_gray, lines, 1, CV_PI/180, 50, 50, 10
                );
    
    //Get the lines around the center
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        if(segmentIntersectRectangle(l[0], l[1], l[2], l[3],src)){
            //Add line's points to the data to prepare for kmeans
            addPointTo(l, &matPoint);
            validLines.push_back(l);
            
        }
        
    }
    cv::Mat labels;
    cv::Mat centers(2, 1, CV_32FC2);
    
    
    //Find 2 centers
    cv::kmeans(matPoint, 2, labels,
               cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, 0.01),
               3, cv::KMEANS_PP_CENTERS, centers);
    
    
    
    
    vector<Vec4i> firstLines;
    vector<Vec4i> secondLines;
    //Distribute lines according to centers
    Vec2f firstCenter = centers.at<Vec2f>(0,0);
    for( size_t i = 0; i < validLines.size(); i++ )
    {
        Vec4i l = validLines[i];
        if (checkIfInTheRange(l, 5, firstCenter[0],firstCenter[1])) {
            firstLines.push_back(l);
            
        }else{
            secondLines.push_back(l);
        }
        
    }
    int longestLineInFirst = findMaxLengthIndex(firstLines);
    int longestLineInSecond = findMaxLengthIndex(secondLines);
    
    Vec4i firstLine = firstLines.at(longestLineInFirst);
    Vec4i secondLine = secondLines.at(longestLineInSecond);
    
    Vec4i minuteHand,hourHand;
    
    if (getLength(firstLine)>getLength(secondLine)) {
        minuteHand = firstLine;
        hourHand = secondLine;
    }else{
        minuteHand = secondLine;
        hourHand = firstLine;
    }
    //Find the intersection of lines
    Point center = getIntersectionOfLines(minuteHand, hourHand);
    
   
    //Find the dominant sides of those lines
    Point minutePoint = getDominantCordinate(minuteHand, center);
    Point hourPoint =  getDominantCordinate(hourHand, center);
    
    
    //Calculate time
    float angleMinute = fmod(360 + RAD*atan2(minutePoint.x-center.x, center.y - minutePoint.y),360);
    float angleHour = fmod(360 + RAD*atan2(hourPoint.x-center.x, center.y - hourPoint.y),360);
    int minute = round(angleMinute/6);
    int hour = angleHour/30;
    string result = "It is " + to_string(hour) + " : " + to_string(minute);
    
    
    
    
    
   
    return result;
    
}



bool checkIfInTheRange(Vec4i l,int thresh,int centerX,int centerY){
    //Point to line distance claculation
    double normalLength = hypot(l[2] - l[0], l[3] - l[1]);
    double distance = (double)((centerX - l[0]) * (l[3] - l[1]) - (centerY - l[1]) * (l[2] - l[0])) / normalLength;
    return thresh > abs(distance);
    
    
}
void addPointTo(Vec4i l, Mat *matPoint){
    //Adding points to line
    int startX = l[0] < l[2] ? l[0]: l[2];
    int endX = l[0] > l[2] ? l[0]: l[2];
    int startY = l[1] < l[3] ? l[1]: l[3];
    int endY = l[1] > l[3] ? l[1]: l[3];
    double slope = ((double)(l[1] - l[3])/(double)(l[0]-l[2]));
    int intercept = l[1]-slope*l[0];
    if(slope ==INFINITY){
        for (int y = startY; y <= endY ; y++) {
            matPoint->push_back(Vec2f(startX,y));
            
        }
        
    }
    if (endX-startX > endY-startY) {
        
        for (int x = startX; x <= endX ; x++) {
            matPoint->push_back(Vec2f(x,(slope*x+intercept)));
            
        }
    }else{
        for (int y = startY; y <= endY ; y++) {
            matPoint->push_back(Vec2f(((y-intercept)/slope),y));
            
        }
        
    }
    
}
int findMaxLengthIndex(vector<Vec4i> lines){
    double max = 0;
    int index = 0;
    for (int i = 0; i<lines.size(); i++) {
        Vec4i l =lines.at(i);
        double length = getLength(l);
        if(length>max){
            max = length;
            index = i;
        }
    }
    return index;
}
double getLength(Vec4i l){
    return sqrt(pow(l[0]-l[2],2.0)+ pow(l[1]-l[3],2.0));
}

Point2f getIntersectionOfLines(Vec4i line1, Vec4i line2)
{
    
    double a1 = (line1[1] - line1[3]) / (double)(line1[0]- line1[2]);
    double b1 = line1[1] - a1 * line1[0];
    
    double a2 = (line2[1] - line2[3]) / (double)(line2[0]- line2[2]);
    double b2 = line2[1] - a2 * line2[0];
    
    if (abs(a1 - a2) < 1e-8)
        cout <<  "error";
    
    double x = (b2 - b1) / (a1 - a2);
    double y = a1 * x + b1;
    return *new Point2f(x, y);
}

Point getDominantCordinate(Vec4i line, Point p)
{
    double power1 = hypot(line[0]-p.x, line[1] - p.y);
    double power2 =hypot(line[2]-p.x, line[3] - p.y);
    if(power1>power2)
        return Point(line[0],line[1]);
    else
        return Point(line[2],line[3]);
    
}

bool segmentIntersectRectangle(
                               double a_p1x,
                               double a_p1y,
                               double a_p2x,
                               double a_p2y, Mat src)
{
    // Find min and max X for the segment
    double a_rectangleMinX = src.cols/2 -20;
    double a_rectangleMinY =src.rows/2 -20;
    double a_rectangleMaxX = src.cols/2 +20;
    double a_rectangleMaxY=src.rows/2 + 20;
    double minX = a_p1x;
    double maxX = a_p2x;
    
    if(a_p1x > a_p2x)
    {
        minX = a_p2x;
        maxX = a_p1x;
    }
    
    // Find the intersection of the segment's and rectangle's x-projections
    
    if(maxX > a_rectangleMaxX)
    {
        maxX = a_rectangleMaxX;
    }
    
    if(minX < a_rectangleMinX)
    {
        minX = a_rectangleMinX;
    }
    
    if(minX > maxX) // If their projections do not intersect return false
    {
        return false;
    }
    
    // Find corresponding min and max Y for min and max X we found before
    
    double minY = a_p1y;
    double maxY = a_p2y;
    
    double dx = a_p2x - a_p1x;
    
    if(abs(dx) > 0.0000001)
    {
        double a = (a_p2y - a_p1y) / dx;
        double b = a_p1y - a * a_p1x;
        minY = a * minX + b;
        maxY = a * maxX + b;
    }
    
    if(minY > maxY)
    {
        double tmp = maxY;
        maxY = minY;
        minY = tmp;
    }
    
    // Find the intersection of the segment's and rectangle's y-projections
    
    if(maxY > a_rectangleMaxY)
    {
        maxY = a_rectangleMaxY;
    }
    
    if(minY < a_rectangleMinY)
    {
        minY = a_rectangleMinY;
    }
    
    if(minY > maxY) // If Y-projections do not intersect return false
    {
        return false;
    }
    
    return true;
}
