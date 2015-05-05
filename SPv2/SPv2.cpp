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
* Date Modified: May 1, 2015
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
#define LOAD_TEST_DEST "C:\\Users\\Abigail_pc\\Documents\\SP\\SP Data Set\\Handwritten\\save\\characters by single person\\"
#define TOTAL_NUMBER_INDEX 100
#define TOTAL_NUMBER_INDEX_CHAR 700
#define NUMBER_OF_IMAGES_PER_ALPHABET_CHARACTER 9 //# of test image loaded
#define NUMBER_OF_CHARACTERS 71

/**
* Global Variables
*/
Mat img;
int imgWidth, imgHeight;
ofstream myfile("example.odt"); 

//mainly for caption detection
vector<vector<Point> > contours;
int areaWholeWhite[TOTAL_NUMBER_INDEX];
Mat captionMask[TOTAL_NUMBER_INDEX];
int captionCount = 0, correctCaptionsCount = 0, characterCount = 0;
RotatedRect captionsEllipse[TOTAL_NUMBER_INDEX];
RotatedRect captionsDetected[TOTAL_NUMBER_INDEX];
int averageBalloonWidth = 0, averageBalloonHeight = 0;

//mainly for character detection
Rect characterDetection[TOTAL_NUMBER_INDEX_CHAR];
int averageCharWidth = 0, averageCharHeight = 0;
Mat extractedChar[TOTAL_NUMBER_INDEX_CHAR];

//mainly for character recognition
String alphabet[2] = { "Hiragana", "Katakana" };
String characters[71] = {
	"a", "i", "u", "e", "o",
	"ka", "ki", "ku", "ke", "ko",
	"sa", "shi", "su", "se", "so",
	"ta", "ti", "tsu", "te", "to",
	"na", "ni", "nu", "ne", "no",
	"ha", "hi", "fu", "he", "ho",
	"ma", "mi", "mu", "me", "mo",
	"ya", "yu", "yo",
	"ra", "ri", "ru", "re", "ro",
	"n", "wo", "wa",
	"ga", "gi", "gu", "ge", "go",
	"za", "ji", "zu", "ze", "zo",
	"da", "di", "zu", "de", "do",
	"ba", "bi", "bu", "be", "bo",
	"pa", "pi", "pu", "pe", "po"
};

// mainly for file
ifstream readFile("file.txt");


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
}// end of replaceROIWithOrigImage

bool sortByXY(RotatedRect c, RotatedRect d){
	return c.center.x > d.center.x;
	return c.center.y < d.center.y;
}// end of sortByXY

bool sortByXYRect(Rect c, Rect d){

	if (c.tl().x == d.tl().x)
		return c.tl().y < d.tl().y;
	if (c.tl().y == d.tl().y)
		return c.tl().x > d.tl().x;

	return c.tl().x > d.tl().x;
	return c.tl().y < d.tl().y;
	
	
	
}// end of sortByXY

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
			
			captionsEllipse[captionCount] = minEllipse[i];


			averageBalloonWidth = averageBalloonWidth + minEllipse[i].size.width;
			averageBalloonHeight = averageBalloonHeight + minEllipse[i].size.height;
			
			captionCount++;
		}
	}
	averageBalloonHeight /= captionCount;
	averageBalloonWidth /= captionCount;

	
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

Mat CaptionDetection(Mat inputImage){
	Mat outputImage, binaryImage, captionDetectImage;

	outputImage = Mat::zeros(inputImage.size(), CV_8UC3);
	outputImage = binarizeImage(outputImage);

	binaryImage = captionDetectImage = binarizeImage(inputImage);
	threshold(captionDetectImage, captionDetectImage, 224, 250, 0); //IJIP-290-libre.pdf

	GaussianBlur(captionDetectImage, captionDetectImage, Size(9, 9), 0, 0);
	captionDetectImage = fittingEllipse(0, 0, captionDetectImage);

	for (int i = 0; i < captionCount; i++){
		Mat replacedImg = replaceROIWithOrigImage(inputImage.clone(), captionMask[i]);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(10, 12));
		Mat binarizedReplaceImg = binarizeImage(replacedImg);
		Mat erodeImage;

		erode(binarizedReplaceImg, erodeImage, element1);
		
		float area = countNonZero(erodeImage);
		int areaBlack = areaWholeWhite[i] - area;
		float whitePercent = (area / (captionsEllipse[i].size.width * captionsEllipse[i].size.height)) * 100;
		
		if (whitePercent <= 55 && whitePercent >=35	){
			captionsDetected[correctCaptionsCount] = captionsEllipse[i];
			ellipse(outputImage, captionsEllipse[i], Scalar(255, 255, 255), -1, 8);
			correctCaptionsCount++;
		}
	}
	
	sort(captionsDetected, captionsDetected + correctCaptionsCount, sortByXY);

	return replaceROIWithOrigImage(inputImage.clone(), outputImage);
} // end of CaptionDetection

/**
* Character Detection/Extraction Functions:
*	CharacterDetection
*/
Mat CharacterDetection(Mat inputImage){
	Mat outputImage, binaryImage, cleanImage, erodeImage;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	binaryImage = binarizeImage(inputImage);

	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 12)); //6,3

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


	// Draw polygonal contour + bonding rects 
	Mat drawing = Mat::zeros(erodeImage.size(), CV_8UC3);
	Mat outout = inputImage;
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);

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
		Scalar color = Scalar::all(255);
		int whitechar = countNonZero(binarizeImage(characterImg)),
			totalpixelchar = boundRect[i].width * boundRect[i].height,
			blackchar = totalpixelchar - whitechar;

		if (boundRect[i].width <= inputImage.cols / 20 && boundRect[i].width >= inputImage.cols / 50){
			averageCharWidth += boundRect[i].width;
			averageCharHeight += boundRect[i].height;

			int sizechar = 0, charheight = 21;

			while (sizechar + charheight <= boundRect[i].height) {
				Point start(1, sizechar);
				Point endPoint(boundRect[i].width, sizechar + charheight);
				Mat characterIndividual;

				characterDetection[characterCount] = Rect(boundRect[i].tl().x + 1, boundRect[i].tl().y + sizechar, boundRect[i].width, charheight);

				characterCount++;
				sizechar += 20;
			}
		}//end of if

	}// end of for

			averageCharWidth = averageCharWidth / characterCount;
			averageCharHeight = averageBalloonHeight / characterCount;

			cout << averageCharWidth << "\t" << averageCharHeight << endl;

			sort(characterDetection, characterDetection + characterCount, sortByXYRect);

			imwrite((String)SAVE_FILE_DEST + "outout.jpg", outout);

			return drawing;
} // end of CharacterDetection

void characterIndv(Mat inputImage){
	for (int i = 0; i < characterCount; i++){
		Mat drawing = Mat::zeros(inputImage.size(), CV_8UC3);
		rectangle(drawing, characterDetection[i], Scalar::all(255), -1, 8);
		extractedChar[i] = replaceROIWithOrigImage(inputImage.clone(), binarizeImage(drawing));
		//imwrite((string)SAVE_FILE_DEST + "char[" + to_string(i) + "].jpg", extractedChar[i]);
	}
}

/**
* Character Recognition Functions:
*	CharacterFeatureExtraction
*	CharacterRecognition
*/

/*
TODO: Plot the values in image then use that to compare
*/

Mat CharacterFeatureExtraction(){
	Mat plottedImg(1500, 1500, CV_8UC3, Scalar(0, 0, 0));
	int r, g, b;
	String imageFileName;
	Mat loadImg, loadImg2;
	int plotNumber = 0;
	Point plottedXY[2200];

	for (int k = 0; k < 1/*2*/; k++){
		for (int i = 0; i < 15; i++){ //number of characters tested

			r = (rand() *(i + 1)) % 255;
			g = (rand() *(i + 1)) % 255;
			b = (rand() *(i + 1)) % 255;

			for (int j = 1; j <= NUMBER_OF_IMAGES_PER_ALPHABET_CHARACTER; j++){
				imageFileName = alphabet[k] + "\\" + characters[i] + " (" + to_string(j) + ").jpg";
				loadImg = imread((string)LOAD_TEST_DEST + imageFileName, 1);

				if (loadImg.data){

					loadImg = binarizeImage(loadImg);

					Mat element1 = getStructuringElement(MORPH_RECT, Size(50, 20)); //10,5
					erode(loadImg, loadImg2, element1);

					vector<vector<Point> > contours;
					vector<Vec4i> hierarchy;

					/// Find contours
					findContours(loadImg2, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

					/// Approximate contours to polygons + get bounding rects and circles
					vector<vector<Point> > contours_poly(contours.size());
					vector<Rect> boundRect(contours.size());

					for (int l = 0; l < contours.size(); l++)
					{
						approxPolyDP(Mat(contours[l]), contours_poly[l], 3, true);
						boundRect[l] = boundingRect(Mat(contours_poly[l]));

						if (contours.size() == 1) loadImg2 = loadImg(boundRect[l]);
						else if (contours.size() > 1 && l == 1) loadImg2 = loadImg(boundRect[l]);
					}

					resize(loadImg2.clone(), loadImg2, Size(150, 150), 0, 0, INTER_LINEAR);

					element1 = getStructuringElement(MORPH_RECT, Size(10, 5));
					erode(loadImg2, loadImg2, element1);

					int height = loadImg2.cols,
						width = loadImg2.rows,
						widthHeightStrip = 20,
						numberPixelHorizontal = width*widthHeightStrip,
						numberPixelVertical = height*widthHeightStrip,
						x = loadImg2.rows/2,
						y = loadImg2.cols/2,
						featureX, featureY;

					Mat verticalStrip = loadImg2(Rect(x, 0, widthHeightStrip, width)),
						horizontalStrip = loadImg2(Rect(0, y, height, widthHeightStrip));

					imwrite((String)SAVE_FILE_DEST + characters[i] + " (" + to_string(j) + ").jpg", horizontalStrip);

					featureX = numberPixelHorizontal - countNonZero(horizontalStrip);
					featureY = numberPixelVertical - countNonZero(verticalStrip);

					

					Point featuresTopPixel, featuresBottomPixel;
					featuresTopPixel.x = featureX - 100;
					featuresTopPixel.y = featureY - 100;
					featuresBottomPixel.x = featuresTopPixel.x;
					featuresBottomPixel.y = featuresTopPixel.y;

					rectangle(plottedImg, featuresTopPixel, featuresBottomPixel, Scalar(b, g, r), 3);

					myfile << characters[i] << "(" << j << ")" << "\t" << featureX << "\t" << featureY << endl;

					plotNumber++;
				}
			}
			myfile << endl;
		}
	}
	cout << plotNumber << endl;
	return plottedImg;
}

Mat MatchingMethod(int, void*, Mat inputImage)
{

	int index = 10;
	inputImage = extractedChar[index];
	Mat mask = inputImage;
	Mat img_display, result;
	inputImage.copyTo(img_display);


	Mat templ = imread("C:\\Users\\Abigail_pc\\Documents\\SP\\SP Data Set\\Hiragana\\hiragana\\re (2).jpg");

	//process inputImage
	/*Mat erodeImage, erodeTemplate;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3)); //6,3
	erode(inputImage, erodeImage, element1);
	erode(templ, erodeTemplate, element1);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(binarizeImage(erodeImage), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	int indexBounding;
	if (contours.size() > 1) indexBounding = contours.size()-1;
	else indexBounding = 0;
	Scalar color = Scalar::all(0);
	inputImage = inputImage(boundRect[indexBounding]);
	*/
	//process template
	Mat erodeTemplate;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(10, 15)); //6,3
	erode(templ, erodeTemplate, element1);

	int indexBounding;
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;
	findContours(binarizeImage(erodeTemplate), contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly2(contours2.size());
	vector<Rect> boundRect2(contours2.size());

	for (int i = 0; i < contours2.size(); i++)
	{
		approxPolyDP(Mat(contours2[i]), contours_poly2[i], 3, true);
		boundRect2[i] = boundingRect(Mat(contours_poly2[i]));
	}

	Mat contoursImg = templ;
	indexBounding = 0;
	for (int i = 0; i< contours2.size(); i++)
	{
		Scalar color = Scalar(255, 0, 0);
		float whitepercentageContour = ((float)countNonZero(binarizeImage(erodeTemplate(boundRect2[i]))) / (contoursImg(boundRect2[i]).cols*contoursImg(boundRect2[i]).rows)) * 100;
		printf("%d\t%f\n", i,whitepercentageContour);
		if (whitepercentageContour <= 50){
			indexBounding = i;
			printf("indexBounding: %d\n", indexBounding);
			rectangle(contoursImg, boundRect2[i].tl(), boundRect2[i].br(), color, 2, 8, 0);
			break;
		}
	}

	templ = templ(boundRect2[indexBounding]);
	imwrite((String)SAVE_FILE_DEST + "templateRect.jpg", templ);
	
	//Retain Aspect Ratio
	// from http://opencv-users.1802565.n2.nabble.com/Correct-way-to-quot-resize-quot-an-image-Mat-w-o-altering-aspect-ratio-td7580627.html
	Size newSize(21,21);
	float ratios[2] = { (float)newSize.width / templ.cols,
		(float)newSize.height / templ.rows };
	Size sizeIntermediate(templ.cols, templ.rows);
	if (ratios[0] < ratios[1])
	{
		sizeIntermediate.width = (int)(sizeIntermediate.width * ratios[0] + 0.5);
		sizeIntermediate.height = (int)(sizeIntermediate.height * ratios[0] + 0.5);
	}
	else
	{
		sizeIntermediate.width = (int)(sizeIntermediate.width * ratios[1] + 0.5);
		sizeIntermediate.height = (int)(sizeIntermediate.height * ratios[1] + 0.5);
	}

	cout << sizeIntermediate << endl;
	resize(templ, templ, sizeIntermediate);

	/*Mat element2 = getStructuringElement(MORPH_RECT, Size(1, 1));
	dilate(inputImage, inputImage, element2);
	dilate(templ, templ, element2);
	erode(inputImage, inputImage, element2);
	erode(templ, templ, element2);

	inputImage = binarizeImage(inputImage);
	templ = binarizeImage(templ);*/

	imwrite((String)SAVE_FILE_DEST + "template.jpg", templ);
	imwrite((String)SAVE_FILE_DEST + "templateInput.jpg", contoursImg);

	/*template matching*/
	
	/// Create the result matrix
	int result_cols = inputImage.cols - templ.cols + 1;
	int result_rows = inputImage.rows - templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(inputImage, templ, result, 5);
	//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	matchLoc = maxLoc;

	/// Show me what you got
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0,0,255), 1, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	//putText(mask, ".", matchLoc, FONT_HERSHEY_PLAIN, 0.5, Scalar(0, 255, 0), 1, 8);
	imwrite((String)SAVE_FILE_DEST + "templateresult.jpg", result);

	Mat b = binarizeImage(mask(Rect(matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows))));
	float whitepercentage = ((float)countNonZero(b) / (templ.cols * templ.rows)) * 100;
	cout << maxVal <<"\t" << whitepercentage<< "%" << endl;

	//cout << maxVal << endl;
	//cout << matchLoc << "\t" << characterDetection[index].tl() << "\t" << matchLoc.x + templ.cols << "\t" << matchLoc.y<< endl;

	if (whitepercentage >= 60)
		cout << "Template found" << endl;
	else
		cout << "Template NOT found" << endl;

	return img_display;
}



/**
* Put to text 
*/
Mat printToImg (Mat inputImg) {

	String line;
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1.5;
	int thickness = 2;
	Point textOrg(10, 130);

	for (int i = 0; i < captionCount && !readFile.eof(); i++){
		getline(readFile, line);
		ellipse(inputImg, captionsDetected[i], Scalar::all(255), -1, 8);
		putText(inputImg, line, Point(captionsDetected[i].center.x - 30, captionsDetected[i].center.y), fontFace, fontScale, Scalar::all(0), thickness, 8);
	}

	return inputImg;
}

Mat printToCharacter(Mat inputImg) {

	String line;
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;

	cout << characterCount << endl;

	for (int i = 0; i < captionCount; i++){
		
		ellipse(inputImg, captionsDetected[i], Scalar::all(255), -1, 8);
		
	}

	for (int i = 0; i < characterCount && !readFile.eof(); i++){
		getline(readFile, line);
		//rectangle(inputImg, characterDetection[i].tl(), characterDetection[i].br(), Scalar::all(255), -1, 8);
		putText(inputImg, line, characterDetection[i].tl(), fontFace, fontScale, Scalar::all(0), thickness, 8);
		
	}

	return inputImg;
}

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
	Mat origImg = img;
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

	characterIndv(origImg);

	/*img = CharacterFeatureExtraction();
	imwrite((string)SAVE_FILE_DEST + "Character_Plot.jpg", img);*/

	imwrite((String)SAVE_FILE_DEST + "PUTTEXT.jpg", printToCharacter(img));

	imwrite((String)SAVE_FILE_DEST + "MATCH.jpg", MatchingMethod(0,0, origImg));
	
}