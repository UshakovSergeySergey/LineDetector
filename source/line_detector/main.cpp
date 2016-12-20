#include "stdafx.h"
#include "line_detector.h"

int main()
{
	cv::Mat image = cv::imread("test.jpg");
	const double scale = 0.25f;
	cv::resize(image, image, cv::Size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale)), 0, 0, cv::INTER_AREA);

	LineDetector line_detector;
	line_detector.detect(image);
	return 0;
}

//#include "stdafx.h"
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ximgproc.hpp>
//#include <iostream>
//#include <ctime>
//#include <limits>
//#include <numeric>
//#include <array>
//
//float dist(const cv::Point2i& p1, const cv::Point2i& p2)
//{
//	cv::Point p = p1 - p2;
//	return sqrt(static_cast<float>(p.x * p.x + p.y * p.y));
//}
//
//void splitLinear(const std::vector<std::vector<cv::Point2i>>& lines, std::vector<std::vector<cv::Point2i>>& segments)
//{
//	segments.clear();
//	for each(auto& line in lines)
//	{
//		std::vector<std::vector<cv::Point2i>> current_segments;
//		processLine(line, current_segments);
//		if(!current_segments.empty())
//			segments.insert(segments.end(), current_segments.begin(), current_segments.end());
//	}
//	auto segment = segments.begin();
//	while(segment != segments.end())
//	{
//		if(dist(segment->front(), segment->back()) < 20)
//			segment = segments.erase(segment);
//		else
//			++segment;
//	}
//}
//
//int main(int argc, char** argv)
//{
//	cv::Mat image = cv::imread("test.jpg");
//
//	const std::clock_t begin_time = std::clock();
//////////////////find lines////////////////
//	std::cout << "find lines" << std::endl;
//	double scale = 0.25f;//1.0f / std::sqrt(image.cols * image.rows / 3.0f / 500000.0f);//round(std::sqrt(image.size / 3.0f / 500000), 1); == 0.25
//	cv::resize(image, image, cv::Size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale)), 0, 0, cv::INTER_AREA);
//////////////////skeletonize////////////////
//	std::cout << "split linear" << std::endl;
//	std::vector<std::vector<cv::Point2i>> filtered_lines;
//	splitLinear(lines, filtered_lines);
//	std::cout << "number of lines = " << filtered_lines.size() << std::endl;
//	cv::Mat original_image = cv::imread("test.jpg");
//	scale = 0.25f;
//	cv::resize(original_image, original_image, cv::Size(static_cast<int>(original_image.cols * scale), static_cast<int>(original_image.rows * scale)), 0, 0, cv::INTER_AREA);
//	for each(auto& line in filtered_lines)
//	{
//		if(line.size() <= 0)
//			std::cout << "fuck" << std::endl;
//		cv::line(original_image, line.front(), line.back(), cv::Scalar(0, 0, 255), 1);
//		//for each(auto& point in line)
//		//{
//		//	if(point.x < 0 || original_image.cols <= point.x || point.y < 0 || original_image.rows <= point.y)
//		//	{
//		//		std::cout << point.x << " " << point.y << " --- " << original_image.cols << " " << original_image.rows << std::endl;
//		//		continue;
//		//	}
//		//	original_image.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(0, 0, 255);
//		//}
//	}
//	std::cout << "total = " << float(std::clock() - begin_time) / static_cast<float>(CLOCKS_PER_SEC) << std::endl;
//	cv::imwrite("result.png", original_image);
//
//	return 0;
//}
