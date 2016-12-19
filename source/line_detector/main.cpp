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
//float distanceToLine(const cv::Point2i& point, const std::array<float, 4>& line_equation)
//{
//	return (line_equation[0] * point.x + line_equation[1] * point.y + line_equation[2]) / line_equation[3];
//}
//
//bool isInCorridor(const cv::Point2i& point, const std::array<float, 4>& line_equation, const float corridor_width = 4.0f)
//{
//	return abs(distanceToLine(point, line_equation)) <= corridor_width;
//}
//
//void lineEquation(const cv::Point2f& p1, const cv::Point2f& p2, std::array<float, 4>& line_equation)
//{
//	float a = p1.y - p2.y;
//	float b = p2.x - p1.x;
//	float c = p1.x * p2.y - p2.x * p1.y;
//	float sqr = sqrt(a * a + b * b);
//	line_equation[0] = a;
//	line_equation[1] = b;
//	line_equation[2] = c;
//	line_equation[3] = sqr;
//}
//
//void fitLine(const std::vector<cv::Point2i>& points, std::array<float, 4>& line_equation)
//{
//	cv::Vec4f line;
//	cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
//	const cv::Point2f p1(line(2), line(3));
//	const cv::Point2f p2(line(2) + 10.0 * line(0), line(3) + 10.0f * line(1));
//	lineEquation(p1, p2, line_equation);
//}
//
//float adjustLinearity(const float linearity, const std::vector<cv::Point2i>& segment)
//{
//	if(segment.size() <= 20)
//		return linearity / 0.5f;
//	if(20 < segment.size() && segment.size() <= 100)
//		return linearity / (0.5 + 0.5f * (segment.size() - 20) / 80.0f);
//	return linearity;
//}
//
//float segmentLinearity(const std::vector<cv::Point2i>& line)
//{
//	std::vector<cv::Point2f> segment;
//	segment.reserve(line.size());
//	for each(auto point in line)
//	{
//		segment.push_back(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
//	}
//
//	const cv::Point2f p1 = segment.front();
//	std::for_each(segment.begin(), segment.end(), [&](cv::Point2f& point)
//	{
//		point -= p1;
//		point.x = abs(point.x);
//		point.y = abs(point.y);
//	});
//
//	const cv::Point2f p2 = segment.back();
//	float hypot = sqrt(p2.x * p2.x + p2.y * p2.y);
//	if(hypot <= std::numeric_limits<float>::epsilon())
//		return std::numeric_limits<float>::max();
//
//	const float cos = p2.x / hypot;
//	const float sin = -p2.y / hypot;
//	cv::Matx22f mat;
//	mat << cos, -sin,
//		   sin, cos;
//	std::for_each(segment.begin(), segment.end(),  [&](cv::Point2f& point)
//	{
//		cv::Vec2f vec(point.x, point.y);
//		vec = mat * vec;
//		point = cv::Point2f(vec(0), vec(1));
//	});
//
//	auto first_lambda = [](cv::Point2f sum, cv::Point2f point) -> cv::Point2f
//	{
//		point.x = abs(point.x);
//		point.y = abs(point.y);
//		return sum + point;
//	};
//	const cv::Point2f first_sum = std::accumulate(segment.begin(), segment.end(), cv::Point2f(0.0f, 0.0f), first_lambda);
//	const cv::Point2f second_sum = std::accumulate(segment.begin(), segment.end(), cv::Point2f(0.0f, 0.0f), [](cv::Point2f sum, cv::Point2f point){ return sum + point; });
//	const float sum = (first_sum.y * 0.8f + abs(second_sum.y) * 0.2f) / segment.size();
//	return sum;
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
