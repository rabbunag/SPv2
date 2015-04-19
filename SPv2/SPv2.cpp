/**
* Description: Special Problem: Optical Character Recognition of Hiragana and Katakana in Manga Pages and Its Translation to English Language using Translate API
* Program Version: 2
* Author of Code: Raquel Abigail Buñag
* Author of Paper: Raquel Abigail Buñag and Marie Yvette de Robles
*
* Run File: SP_Binarize <image file>
* Example of <image file>: C:\Users\Abigail_pc\Documents\SP\SP Data Set\Raw Manga\Cardaptor Sakura\Cardcaptor Sakura 09-12\Cardcaptor Sakura 09\Sakura_v09-007.jpg
*
* Date Code Created: July 24, 2014
* Date Version 2 Created: March 7, 2015
* Date Modified: April 19, 2015
*/

/**
*	NOTES:
*		width = number of cols
*		height = number of rows
*/

/**
* Headers
*/
#include "stdafx.h"
#include "cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <ocl\ocl.hpp>

using namespace cv;
using namespace std;

/**
* Definitions
*/
#define SAVE_FILE_DEST "C:\\Users\\Abigail_pc\\Documents\\Github\\SPv2\\Debug\\"
#define TOTAL_NUMBER_INDEX 100

/**
* Global Variables
*/
Mat img;
int imgWidth, imgHeight;
int averageBalloonWidth = 0;
int averageBalloonHeight = 0;
ofstream myfile("example.odt"); 
vector<vector<Point> > contours;
int areaWholeWhite[TOTAL_NUMBER_INDEX];
Mat captionMask[TOTAL_NUMBER_INDEX];
int contourWidth[TOTAL_NUMBER_INDEX];
int contourHeight[TOTAL_NUMBER_INDEX];
int captionCount = 0;
RotatedRect captionsEllipse[TOTAL_NUMBER_INDEX];

/**
*					FUNCTIONS
*/

Mat binarizeImage(Mat inputOutputImage){
	Mat img_gray;

	//grayscale
	cvtColor(inputOutputImage, img_gray, CV_BGR2GRAY);

	//otsu's method
	threshold(img_gray, inputOutputImage, 0, 255, CV_THRESH_OTSU);

	return inputOutputImage;
} // end of binarizeImage


void nameAndSaveImage(char ** filenNameAndDestination, Mat img, char * prefix){
	//naming and saving result
	char * nameAndDestination = "";
	char *next = NULL;
	if (filenNameAndDestination[2] == NULL) {
		char * characterHandler;
		characterHandler = strtok_s(filenNameAndDestination[1], "\\", &next);

		while (characterHandler != NULL){
			nameAndDestination = characterHandler;
			characterHandler = strtok_s(NULL, "\\", &next);
		}

		string binarizeString = prefix;
		string nameAndDestinationString = binarizeString + nameAndDestination;

		imwrite(nameAndDestinationString, img);

	}
	else {
		nameAndDestination = filenNameAndDestination[2];
		imwrite(nameAndDestination, img);
	}
} // end of nameAndSaveImage

/**
* Caption Detection Functions:
*	invertImage
*	fittingEllipes
*	CaptionDetection
*/

// Modified version of thresold_callback function 
// from http://docs.opencv.org/doc/tutorials/imgproc/shapedescriptors/bounding_rotated_ellipses/bounding_rotated_ellipses.html
Mat fittingEllipse(int, void*, Mat inputImage)
{
	Mat outputImage;
	vector<Vec4i> hierarchy;
	int numberOfCaptions = 0;

	// Detect edges using Threshold
	threshold(inputImage, inputImage, 224, 250, THRESH_BINARY);

	findContours(inputImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<RotatedRect> minEllipse(contours.size());
	

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 5)
			minEllipse[i] = fitEllipse(Mat(contours[i]));
	}

	//Draw ellipse/caption
	outputImage = Mat::zeros(inputImage.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);
		Mat drawing = Mat::zeros(inputImage.size(), CV_8UC3);

		ellipse(drawing, minEllipse[i], color, -1, 8);

		drawing = binarizeImage(drawing);
		int area = countNonZero(drawing);

		if (minEllipse[i].size.height >= inputImage.rows / 10 && //IJIP-290-libre.pdf
			minEllipse[i].size.width >= inputImage.cols / 8 && //IJIP-290-libre.pdf
			minEllipse[i].size.height <= inputImage.rows / 3 &&
			minEllipse[i].size.width <= inputImage.cols / 3 &&
			(
			(minEllipse[i].angle >= 0 && minEllipse[i].angle <= 30) ||
			(minEllipse[i].angle >= 80 && minEllipse[i].angle <= 100) ||
			(minEllipse[i].angle >= 170 && minEllipse[i].angle <= 190) ||
			(minEllipse[i].angle >= 260 && minEllipse[i].angle <= 280) ||
			(minEllipse[i].angle >= 350 && minEllipse[i].angle <= 360)
			)){
			ellipse(outputImage, minEllipse[i], color, -1, 8);
			captionMask[captionCount] = drawing;
			areaWholeWhite[captionCount] = area;
			contourHeight[captionCount] = minEllipse[i].size.height;
			contourWidth[captionCount] = minEllipse[i].size.width;
			captionsEllipse[captionCount] = minEllipse[i];


			averageBalloonWidth = averageBalloonWidth + minEllipse[i].size.width;
			averageBalloonHeight = averageBalloonHeight + minEllipse[i].size.height;
			
			captionCount++;
		}

		
		/*if (minEllipse[i].size.height >= inputImage.rows / 8 && //IJIP-290-libre.pdf
			minEllipse[i].size.width >= inputImage.cols / 10 && //IJIP-290-libre.pdf
			minEllipse[i].size.height < inputImage.rows / 3  &&
			minEllipse[i].size.width < inputImage.cols / 3 &&
			(
			(minEllipse[i].angle >= 0 && minEllipse[i].angle <= 10) ||
			(minEllipse[i].angle >= 80 && minEllipse[i].angle <= 100) ||
			(minEllipse[i].angle >= 170 && minEllipse[i].angle <= 190) ||
			(minEllipse[i].angle >= 260 && minEllipse[i].angle <= 280) ||
			(minEllipse[i].angle >= 350 && minEllipse[i].angle <= 360)
			)) {
			//color = Scalar(0, 0, 255);
			averageBalloonWidth = averageBalloonWidth + minEllipse[i].size.width;
			averageBalloonHeight = averageBalloonHeight + minEllipse[i].size.height;
			numberOfCaptions++;
			ellipse(drawing, minEllipse[i], color, -1, 8);
		}*/
	}
	averageBalloonHeight /= captionCount;
	averageBalloonWidth /= captionCount;
	
	//imwrite((string)SAVE_FILE_DEST + "out.jpg", outputImage);
	
	return outputImage;
} // end of fittingEllipse

Mat invertImage(Mat img){
	// get the image data
	int height = img.cols,
		width = img.rows,
		step = img.step;
	unsigned char * input = (unsigned char*)(img.data);

	for (int i = 0; i < imgWidth; i++)
	{
		for (int j = 0; j < imgHeight; j++)
		{
			input[step * i + j] = 255 - input[step * i + j]; // b 
			input[step * i + j + 1] = 255 - input[step * i + j + 1]; // g 
			input[step * i + j + 2] = 255 - input[step * i + j + 2]; // r
		}
	}
	return img;
} // end of invertImage

Mat replaceROIWithOrigImage(Mat inputImg, Mat mask){
	Mat outputImage = inputImg;
	Mat maskImg = mask;
	
	for (int i = 0; i < inputImg.rows; i++) {
		for (int j = 0; j < inputImg.cols; j++) {
			
			if (maskImg.at<uchar>(i, j) == 0) {
				inputImg.at<Vec3b>(i, j)[0] = inputImg.at<Vec3b>(i, j)[1] = inputImg.at<Vec3b>(i, j)[2] = 0;
			}
			
		}
	}
	
	return inputImg;
}

Mat CaptionDetection(Mat inputImage){
	Mat outputImage, binaryImage, captionDetectImage;

	outputImage = Mat::zeros(inputImage.size(), CV_8UC3);
	outputImage = binarizeImage(outputImage);

	binaryImage = captionDetectImage = binarizeImage(inputImage);
	threshold(captionDetectImage, captionDetectImage, 224, 250, 0); //IJIP-290-libre.pdf

	GaussianBlur(captionDetectImage, captionDetectImage, Size(9, 9), 0, 0);
	captionDetectImage = fittingEllipse(0, 0, captionDetectImage);

	//cout << averageBalloonWidth << " X " << averageBalloonHeight << endl;

	//cout << "i" << ")\t" << "WxH" << "\t" << "white%" << "\t" << "areaW" << endl;

	for (int i = 0; i < captionCount; i++){
		//cout << "--------------------------------------------------------------------------------" << endl;
		Mat replacedImg = replaceROIWithOrigImage(inputImage.clone(), captionMask[i]);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(10, 12));
		Mat binarizedReplaceImg = binarizeImage(replacedImg);
		Mat erodeImage;

		erode(binarizedReplaceImg, erodeImage, element1);
		
		float area = countNonZero(erodeImage);
		int areaBlack = areaWholeWhite[i] - area;

		float whitePercent = (area / (contourWidth[i] * contourHeight[i])) * 100;
		
		//cout << i << ")\t" << contourWidth[i] << "X" << contourHeight[i] << "\t" << whitePercent << "\t" << area << endl;
		
		if (whitePercent <= 55 && whitePercent >=35	){
			ellipse(outputImage, captionsEllipse[i], Scalar(255, 255, 255), -1, 8);
		}

		//imwrite((string)SAVE_FILE_DEST + "maskafter[" + to_string(i) + "].jpg", erodeImage);
		
	}
	
	return replaceROIWithOrigImage(inputImage.clone(), outputImage);
} // end of CaptionDetection

/**
* Character Detection Functions:
*	CharacterDetection
*/
Mat CharacterDetection(Mat inputImage){
	Mat outputImage, binaryImage, cleanImage, erodeImage;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	binaryImage = binarizeImage(inputImage);

	Mat element1 = getStructuringElement(MORPH_RECT, Size(6, 3));

	erode(binaryImage, erodeImage, element1);
	imwrite((string)SAVE_FILE_DEST + "Character_Erode.jpg", erodeImage);

	findContours(erodeImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(erodeImage.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255,255,255);
		if (boundRect[i].width <= inputImage.cols / 20 && boundRect[i].width >= inputImage.cols / 50){
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, -1, 8);
		}
		
	}

	drawing = replaceROIWithOrigImage(inputImage, binarizeImage(drawing));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(6, 3));

	Mat erodeImageChar;
	erode(drawing, erodeImageChar, element2);

	for (int i = 0; i< contours.size(); i++)
	{
		Mat characterImg = erodeImageChar(boundRect[i]);
		Scalar color = Scalar(255, 255, 255);
		int whitechar = countNonZero(binarizeImage(characterImg)), 
			totalpixelchar = boundRect[i].width * boundRect[i].height,
			blackchar = totalpixelchar - whitechar;

		if (boundRect[i].width <= inputImage.cols / 20 && boundRect[i].width >= inputImage.cols / 50 && blackchar >= totalpixelchar/2){
			//cout << i << "\t" << whitechar << "\t" << totalpixelchar << "\t" <<  blackchar << endl;
			imwrite((String)SAVE_FILE_DEST + "char_mask[" + to_string(i) + "].jpg", inputImage(boundRect[i]));
		}

	}

	return drawing;
} // end of CharacterDetection

/**
* MAIN FUNCTION
*/
int main(int argc, char** argv)
{

	//check for image file
	if (argv[1] == NULL) {
		printf("Include an image file");
		return 0;
	}
	
	//read image
	img = imread(argv[1], 1);
	imgWidth = img.rows;
	imgHeight = img.cols;

	//binarize image
	imwrite((string)SAVE_FILE_DEST + "Binarize.jpg", binarizeImage(img));
	//nameAndSaveImage(argv, binarizeImage(img), "Binarize_");

	//caption detection
	img = CaptionDetection(img);
	imwrite((string)SAVE_FILE_DEST + "Caption_Detection.jpg", img);

	//character detection
	img = CharacterDetection(img);
	imwrite((string)SAVE_FILE_DEST + "Character_Detection.jpg", img);
	
}