#include "stdafx.h"
#include "line_detector.h"
#include <ctime>
#include <iostream>

int main()
{
	cv::Mat image = cv::imread("test.jpg");
	const double scale = 0.25f;
	cv::resize(image, image, cv::Size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale)), 0, 0, cv::INTER_AREA);

	clock_t begin = clock();
	LineDetector line_detector;
	std::vector<LineDetector::Segment> lines;
	line_detector.detect(image, lines);
	clock_t end = clock();
	std::cout << (end - begin) / static_cast<float>(CLOCKS_PER_SEC) << std::endl;
	return 0;
}
