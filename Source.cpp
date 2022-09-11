#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;

Mat Originalimg, imgThresh, imgGray, imgBlur, imgCanny, imgDil, imgWarp, imgCrop;
vector<Point>initialPoints, finalPoints;
float w = 420, h = 596;

Mat preprocessing(Mat img) {
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);


	return imgDil;
}

vector<Point> getContours(Mat img) {
	vector<vector<Point>> Contours;
	vector<Vec4i> hierarchy;

	findContours(img, Contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(Contours.size());
	vector<Rect> boundRect(Contours.size());

	vector<Point> biggest;
	int maxArea = 0;

	for (int i = 0; i < Contours.size(); i++) {
		int area = contourArea(Contours[i]);
		string objectType;

		if (area > 1000) {
			float Peri = arcLength(Contours[i], true);
			approxPolyDP(Contours[i], conPoly[i], 0.02 * Peri, true);

			if (area > maxArea && conPoly[i].size() == 4) {
				//drawContours(Originalimg, conPoly, i, Scalar(255, 0, 255), 5);
				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
			}
		}
	}
	return biggest;
}



vector<Point> reorder(vector<Point> points)
{
	vector<Point> newPoints;
	vector<int>  sumPoints, subPoints;

	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // 0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //3

	return newPoints;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
	Point2f src[4] = { points[0],points[1],points[2],points[3] };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	return imgWarp;
}

int main() {
	String path = "unScannedImg/img1.jpeg";
	Originalimg = imread(path);
	//resize(Originalimg, Originalimg, Size(), 0.2, 0.2);

	// preprocessing
	imgThresh = preprocessing(Originalimg);

	//Get Contours
	initialPoints = getContours(imgThresh);


	// reorder the points
	finalPoints = reorder(initialPoints);


	// warp the image
	imgWarp = getWarp(Originalimg, finalPoints, w, h);

	//Crop the image
	int cropVal = 2;
	Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
	imgCrop = imgWarp(roi);

	//sharping the image
	Mat sharp_img;
	Mat kernel3 = (Mat_<double>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	filter2D(imgCrop, sharp_img, -1, kernel3, Point(-1, -1), 0, BORDER_DEFAULT);


	//increasing the contrast of image
	Mat contrastImg;
	sharp_img.convertTo(contrastImg, -1, 1.3, 0);
	///type the name of the image you want with extension
	imwrite("scannedimg/imgScanned.jpg", contrastImg);

	//imshow("img", Originalimg);
	//imshow("img warp", imgCrop);
	//imshow("img sharp", sharp_img);
	//imshow("img contrast", contrastImg);
	cout << "image saved" << endl;



	waitKey(0);

	return 0;
}