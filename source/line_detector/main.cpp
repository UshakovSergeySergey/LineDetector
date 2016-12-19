#include "stdafx.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <ctime>
#include <limits>
#include <numeric>
#include <array>

float dist(const cv::Point2i& p1, const cv::Point2i& p2)
{
	cv::Point p = p1 - p2;
	return sqrt(static_cast<float>(p.x * p.x + p.y * p.y));
}

float distanceToLine(const cv::Point2i& point, const std::array<float, 4>& line_equation)
{
	return (line_equation[0] * point.x + line_equation[1] * point.y + line_equation[2]) / line_equation[3];
}

bool isInCorridor(const cv::Point2i& point, const std::array<float, 4>& line_equation, const float corridor_width = 4.0f)
{
	return abs(distanceToLine(point, line_equation)) <= corridor_width;
}

void lineEquation(const cv::Point2f& p1, const cv::Point2f& p2, std::array<float, 4>& line_equation)
{
	float a = p1.y - p2.y;
	float b = p2.x - p1.x;
	float c = p1.x * p2.y - p2.x * p1.y;
	float sqr = sqrt(a * a + b * b);
	line_equation[0] = a;
	line_equation[1] = b;
	line_equation[2] = c;
	line_equation[3] = sqr;
}

void fitLine(const std::vector<cv::Point2i>& points, std::array<float, 4>& line_equation)
{
	cv::Vec4f line;
	cv::fitLine(points, line, cv::DIST_L2, 0.0, 0.01, 0.01);
	const cv::Point2f p1(line(2), line(3));
	const cv::Point2f p2(line(2) + 10.0 * line(0), line(3) + 10.0f * line(1));
	lineEquation(p1, p2, line_equation);
}

float adjustLinearity(const float linearity, const std::vector<cv::Point2i>& segment)
{
	if(segment.size() <= 20)
		return linearity / 0.5f;
	if(20 < segment.size() && segment.size() <= 100)
		return linearity / (0.5 + 0.5f * (segment.size() - 20) / 80.0f);
	return linearity;
}

float segmentLinearity(const std::vector<cv::Point2i>& line)
{
	std::vector<cv::Point2f> segment;
	segment.reserve(line.size());
	for each(auto point in line)
	{
		segment.push_back(cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)));
	}

	const cv::Point2f p1 = segment.front();
	std::for_each(segment.begin(), segment.end(), [&](cv::Point2f& point)
	{
		point -= p1;
		point.x = abs(point.x);
		point.y = abs(point.y);
	});

	const cv::Point2f p2 = segment.back();
	float hypot = sqrt(p2.x * p2.x + p2.y * p2.y);
	if(hypot <= std::numeric_limits<float>::epsilon())
		return std::numeric_limits<float>::max();

	const float cos = p2.x / hypot;
	const float sin = -p2.y / hypot;
	cv::Matx22f mat;
	mat << cos, -sin,
		   sin, cos;
	std::for_each(segment.begin(), segment.end(),  [&](cv::Point2f& point)
	{
		cv::Vec2f vec(point.x, point.y);
		vec = mat * vec;
		point = cv::Point2f(vec(0), vec(1));
	});

	auto first_lambda = [](cv::Point2f sum, cv::Point2f point) -> cv::Point2f
	{
		point.x = abs(point.x);
		point.y = abs(point.y);
		return sum + point;
	};
	const cv::Point2f first_sum = std::accumulate(segment.begin(), segment.end(), cv::Point2f(0.0f, 0.0f), first_lambda);
	const cv::Point2f second_sum = std::accumulate(segment.begin(), segment.end(), cv::Point2f(0.0f, 0.0f), [](cv::Point2f sum, cv::Point2f point){ return sum + point; });
	const float sum = (first_sum.y * 0.8f + abs(second_sum.y) * 0.2f) / segment.size();
	return sum;
}

void processLine(const std::vector<cv::Point2i>& line, std::vector<std::vector<cv::Point2i>>& segments)
{
	const int frame_size = 15;
	const int stride = 2;
	const int rebuild_corridor_each = 10;
	const int min_corridor = 20;

	segments.clear();
	if(line.size() < frame_size)
		return;

	int frame_pointer = 0;
	int step = 0;
	std::vector<cv::Point2i> current_segment;
	std::array<float, 4> corridor;
	corridor[3] = -std::numeric_limits<float>::max();

	while(true)
	{
		bool ok = false;
		std::vector<cv::Point2i> current_frame;
		current_frame.insert(current_frame.end(), std::next(line.cbegin(), frame_pointer), std::next(line.cbegin(), frame_pointer + frame_size));
		if(adjustLinearity(segmentLinearity(current_frame), current_segment) < 1.0f)
		{
			if(current_segment.empty())
			{
				current_segment.insert(current_segment.end(), current_frame.cbegin(), std::prev(current_frame.cend(), stride));
				step = 0;
			}
			if(min_corridor < current_segment.size())
			{
				step++;
				if(corridor[3] < 0)
				{
					fitLine(current_segment, corridor);
					ok = true;
				}
				else
				{
					if(isInCorridor(current_frame.back(), corridor))
						ok = true;
				}
				if(step % rebuild_corridor_each == 0)
					fitLine(current_segment, corridor);
			}
			else
				ok = true;
		}
		if(ok)
			current_segment.insert(current_segment.end(), std::prev(current_frame.cend(), stride), current_frame.cend());
		else
		{
			if(!current_segment.empty())
			{
				segments.push_back(current_segment);
				current_segment.clear();
				corridor[3] = -std::numeric_limits<float>::max();
				frame_pointer += frame_size - stride;
			}
		}
		frame_pointer += stride;
		if(line.size() - frame_size <= frame_pointer)
			break;
	}

	if(!current_segment.empty())
		segments.push_back(current_segment);
}

void splitLinear(const std::vector<std::vector<cv::Point2i>>& lines, std::vector<std::vector<cv::Point2i>>& segments)
{
	segments.clear();
	for each(auto& line in lines)
	{
		std::vector<std::vector<cv::Point2i>> current_segments;
		processLine(line, current_segments);
		if(!current_segments.empty())
			segments.insert(segments.end(), current_segments.begin(), current_segments.end());
	}
	auto segment = segments.begin();
	while(segment != segments.end())
	{
		if(dist(segment->front(), segment->back()) < 20)
			segment = segments.erase(segment);
		else
			++segment;
	}
}

bool nextPoint(cv::Mat& edges, const cv::Point2i& current_point, cv::Point2i& next_point)
{
	std::vector<cv::Point2i> potential_points;
	if(current_point.x < 0 || edges.cols <= current_point.x || current_point.y < 0 || edges.rows <= current_point.y)
		return false;
	if(edges.at<unsigned char>(current_point.y, current_point.x) == 255)
	{
		edges.at<unsigned char>(current_point.y, current_point.x) = 0;
		for(int row = -1; row < 2; row++)
		{
			for(int column = -1; column < 2; column++)
			{
				if(current_point.y + row < 0 || edges.rows <= current_point.y + row || current_point.x + column < 0 || edges.cols <= current_point.x + column)
					continue;
				if(edges.at<unsigned char>(current_point.y + row, current_point.x + column) != 0)
					potential_points.push_back(cv::Point2i(column, row));
			}
		}
		if(potential_points.size() <= 0)
			return false;
		else
		{
			if(potential_points.size() == 1)
			{
				next_point = current_point + potential_points[0];
				return true;
			}
			else
			{
				if(potential_points.size() > 1)
				{
					potential_points.clear();
					for(int row = -1; row < 2; row++)
					{
						for(int column = -1; column < 2; column++)
						{
							if(current_point.y + row < 0 || edges.rows <= current_point.y + row || current_point.x + column < 0 || edges.cols <= current_point.x + column)
								continue;
							if(edges.at<unsigned char>(current_point.y + row, current_point.x + column) == 127)
								potential_points.push_back(cv::Point2i(column, row));
						}
					}
					if(potential_points.size() == 0)
					{
						for(int row = -1; row < 2; row++)
						{
							for(int column = -1; column < 2; column++)
							{
								if(current_point.y + row < 0 || edges.rows <= current_point.y + row || current_point.x + column < 0 || edges.cols <= current_point.x + column)
									continue;
								if(edges.at<unsigned char>(current_point.y + row, current_point.x + column) == 255)
									potential_points.push_back(cv::Point2i(column, row));
							}
						}
					}
					next_point = current_point + potential_points[0];
					return true;
				}
			}
		}
	}
	else
	{
		if(edges.at<unsigned char>(current_point.y, current_point.x) == 127)
		{
			for(int row = -1; row < 2; row++)
			{
				for(int column = -1; column < 2; column++)
				{
					if(current_point.y + row < 0 || edges.rows <= current_point.y + row || current_point.x + column < 0 || edges.cols <= current_point.x + column)
						continue;
					if(edges.at<unsigned char>(current_point.y + row, current_point.x + column) != 255)
						potential_points.push_back(cv::Point2i(column, row));
				}
			}
			if(potential_points.size() == 1)
				edges.at<unsigned char>(current_point.y, current_point.x) = 0;
		}
	}
	return false;
}

void traceSegment(cv::Mat& edges, const cv::Point2i& point, std::vector<cv::Point2i>& segment)
{
	segment.clear();
	segment.push_back(point);

	cv::Point2i next_point;
	cv::Point2i current_point = point;
	while(nextPoint(edges, current_point, next_point))
	{
		current_point = next_point;
		segment.push_back(current_point);
	}
}

void applyFilter(const cv::Mat kernel, cv::Mat& edges)
{
	cv::Mat float_edges;
	edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);
	cv::filter2D(float_edges, float_edges, -1, kernel);
	for(int row = 0; row < edges.rows; row++)
	{
		for(int column = 0; column < edges.cols; column++)
			if(float_edges.at<float>(row, column) == 6)
				edges.at<unsigned char>(row, column) = 0;
	}
}

int main(int argc, char** argv)
{
	cv::Mat image = cv::imread("test.jpg");

	const std::clock_t begin_time = std::clock();
////////////////find lines////////////////
	std::cout << "find lines" << std::endl;
	double scale = 0.25f;//1.0f / std::sqrt(image.cols * image.rows / 3.0f / 500000.0f);//round(std::sqrt(image.size / 3.0f / 500000), 1); == 0.25
	cv::resize(image, image, cv::Size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale)), 0, 0, cv::INTER_AREA);

////////////////clahe////////////////
	std::cout << "clahe" << std::endl;
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	const std::clock_t clahe_time = std::clock();
	for each(auto& channel in channels)
	{
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
		clahe->apply(channel, channel);
	}
	std::cout << "clahe = " << float(std::clock() - clahe_time) / static_cast<float>(CLOCKS_PER_SEC) << std::endl;
	cv::merge(channels, image);

////////////////get skeleton////////////////
	std::cout << "get skeleton" << std::endl;
	const std::clock_t amFilter_time = std::clock();
	cv::ximgproc::amFilter(image, image, image, 3, 0.5, true);
	std::cout << "amFilter = " << float(std::clock() - amFilter_time) / static_cast<float>(CLOCKS_PER_SEC) << std::endl;
	cv::blur(image, image, cv::Size(3, 3));
	cv::Mat edges;
	cv::Canny(image, edges, 200, 300, 5, true);

////////////////skeletonize////////////////
	std::cout << "skeletonize" << std::endl;
	const int kernel_size = 3;
	{
		cv::Mat kernel = (cv::Mat_<double>(kernel_size, kernel_size) <<
			0, 0, -3,
			1, 4,  0,
			0, 1,  0);
		applyFilter(kernel, edges);
	}
	{
		cv::Mat kernel = (cv::Mat_<double>(kernel_size, kernel_size) <<
			-3, 0, 0,
			 0, 4, 1,
			 0, 1, 0);
		applyFilter(kernel, edges);
	}
	{
		cv::Mat kernel = (cv::Mat_<double>(kernel_size, kernel_size) <<
			0, 1,  0,
			1, 4,  0,
			0, 0, -3);
		applyFilter(kernel, edges);
	}
	{
		cv::Mat kernel = (cv::Mat_<double>(kernel_size, kernel_size) <<
			 0, 1, 0,
			 0, 4, 1,
			-3, 0, 0);
		applyFilter(kernel, edges);
	}
	{
		cv::Mat kernel_1 = (cv::Mat_<double>(kernel_size, kernel_size) <<
			-1, -1, -1,
			-1,  9, -1,
			-1, -1, -1);
		cv::Mat float_edges;
		edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);
		cv::filter2D(float_edges, float_edges, -1, kernel_1);
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
				if(0 < float_edges.at<float>(row, column) && float_edges.at<float>(row, column) <= 4)
					edges.at<unsigned char>(row, column) = 0;
		}
		cv::Mat kernel_2 = (cv::Mat_<double>(kernel_size, kernel_size) <<
			-1, 0, -1,
			 0, 5,  0,
			-1, 0, -1);
		cv::Mat res1;
		cv::Mat res2;
		edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);
		cv::filter2D(float_edges, res1, -1, kernel_1);
		cv::filter2D(float_edges, res2, -1, kernel_2);
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
				if(0 < res1.at<float>(row, column) && res1.at<float>(row, column) <= 5 && res2.at<float>(row, column) != 1)
					edges.at<unsigned char>(row, column) = 0;
		}
	}

////////////////trace segments////////////////
	std::cout << "trace segments" << std::endl;
	for(int row = 0; row < edges.rows; row++)
	{
		edges.at<unsigned char>(row, 0) = 0;
		edges.at<unsigned char>(row, edges.cols - 1) = 0;
	}
	for(int column = 0; column < edges.cols; column++)
	{
		edges.at<unsigned char>(0, column) = 0;
		edges.at<unsigned char>(edges.rows - 1, column) = 0;
	}

////////////////junction and endpoint////////////////
	std::cout << "junction and endpoint" << std::endl;
	cv::Mat kernel = (cv::Mat_<double>(kernel_size, kernel_size) <<
			  1,   2,  4,
			128, 256,  8,
			 64,  32, 16);
	cv::Mat float_edges;
	edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);
	cv::filter2D(float_edges, float_edges, -1, kernel);
	for(int row = 0; row < edges.rows; row++)
	{
		for(int column = 0; column < edges.cols; column++)
			if(0 < float_edges.at<float>(row, column) && float_edges.at<float>(row, column) <= 4)
				edges.at<unsigned char>(row, column) = 0;
	}
	cv::Mat edges1;
	cv::cvtColor(edges, edges1, CV_GRAY2BGR);
	for(int row = 0; row < edges.rows; row++)
	{
		for(int column = 0; column < edges.cols; column++)
			if(float_edges.at<float>(row, column) == 256)
			{
				edges.at<unsigned char>(row, column) = 0;
				edges1.at<cv::Vec3b>(row, column) = cv::Vec3b(0, 0, 0);
			}
	}
	{
		float d1 = 256+128+32+4;
		float d2 = 256+128+2+16;
		float d3 = 256+2+8+64;
		float d4 = 256+8+32+1;
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
				if(float_edges.at<float>(row, column) == d1 || float_edges.at<float>(row, column) == d2 || float_edges.at<float>(row, column) == d3 || float_edges.at<float>(row, column) == d4)
				{
					edges.at<unsigned char>(row, column) = 127;
					edges1.at<cv::Vec3b>(row, column) = cv::Vec3b(255, 0, 0);
				}
		}
	}
	{
		float d1 = 256+1+64+8;
		float d2 = 256+1+4+32;
		float d3 = 256+4+16+128;
		float d4 = 256+16+64+2;
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
				if(float_edges.at<float>(row, column) == d1 || float_edges.at<float>(row, column) == d2 || float_edges.at<float>(row, column) == d3 || float_edges.at<float>(row, column) == d4)
				{
					edges.at<unsigned char>(row, column) = 127;
					edges1.at<cv::Vec3b>(row, column) = cv::Vec3b(255, 0, 0);
				}
		}
	}
	{
		float d1 = 256+1+4+16;
		float d2 = 256+4+16+64;
		float d3 = 256+16+64+1;
		float d4 = 256+64+1+4;
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
				if(float_edges.at<float>(row, column) == d1 || float_edges.at<float>(row, column) == d2 || float_edges.at<float>(row, column) == d3 || float_edges.at<float>(row, column) == d4)
				{
					edges.at<unsigned char>(row, column) = 127;
					edges1.at<cv::Vec3b>(row, column) = cv::Vec3b(255, 0, 0);
				}
		}
	}
	{
		float x = 256+1+4+16+64;
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
				if(float_edges.at<float>(row, column) == x)
				{
					edges.at<unsigned char>(row, column) = 127;
					edges1.at<cv::Vec3b>(row, column) = cv::Vec3b(255, 0, 0);
				}
		}
	}
	std::cout << "get endpoints" << std::endl;
	std::vector<cv::Point2i> endpoints;
	for(int row = 0; row < edges.rows; row++)
	{
		for(int column = 0; column < edges.cols; column++)
		{
			if(float_edges.at<float>(row, column) == 288 || float_edges.at<float>(row, column) == 384 || float_edges.at<float>(row, column) == 258 ||
				float_edges.at<float>(row, column) == 264 || float_edges.at<float>(row, column) == 257 || float_edges.at<float>(row, column) == 272 ||
					float_edges.at<float>(row, column) == 320 || float_edges.at<float>(row, column) == 260)
			{
				endpoints.push_back(cv::Point2i(column, row));
			}
		}
	}
	auto lambda_less = [](const cv::Point2i& p1, const cv::Point2i& p2) -> bool
	{
		if(p1.x < p2.x)
			return true;
		return p1.y < p2.y;
	};
	std::sort(endpoints.begin(), endpoints.end(), lambda_less);

	std::cout << "build lines" << std::endl;
	std::vector<std::vector<cv::Point2i>> lines;
	for each(auto& point in endpoints)
	{
		if(edges.at<unsigned char>(point.y, point.x) == 255)
		{
			std::vector<cv::Point2i> segment;
			traceSegment(edges, point, segment);
			lines.push_back(segment);
		}
	}

////////////////////redpoints////////////////////
	std::vector<cv::Point2i> redpoints;
	for(int row = 0; row < edges.rows; row++)
	{
		for(int column = 0; column < edges.cols; column++)
		{
			if(edges.at<unsigned char>(row, column) == 127)
				redpoints.push_back(cv::Point2i(column, row));
		}
	}
	std::sort(redpoints.begin(), redpoints.end(), lambda_less);

	for(int epoch = 0; epoch < 4; epoch++)
	{
		for each(auto redpoint in redpoints)
		{
			int sum = 0;
			for(int row = -1; row < 2; row++)
			{
				for(int column = -1; column < 2; column++)
				{
					if(redpoint.y + row < 0 || edges.rows <= redpoint.y + row || redpoint.x + column < 0 || edges.cols <= redpoint.x + column)
						continue;
					sum += edges.at<unsigned char>(redpoint.y + row, redpoint.x + column);
				}
			}
			if(sum - edges.at<unsigned char>(redpoint.y, redpoint.x) > 0)
			{
				edges.at<unsigned char>(redpoint.y, redpoint.x) = 255;
				std::vector<cv::Point2i> segment;
				traceSegment(edges, redpoint, segment);
				lines.push_back(segment);
			}
		}
	}

	while(true)
	{
		std::vector<int> xx;
		std::vector<int> yy;
		for(int row = 0; row < edges.rows; row++)
		{
			for(int column = 0; column < edges.cols; column++)
			{
				if(edges.at<unsigned char>(row, column) == 255)
				{
					xx.push_back(column);
					yy.push_back(row);
				}
			}
		}
		if(yy.size() <= 0)
			break;
		std::vector<cv::Point2i> segment;
		traceSegment(edges, cv::Point2i(xx[0], yy[0]), segment);
		lines.push_back(segment); 
	}

	std::cout << "split linear" << std::endl;
	std::vector<std::vector<cv::Point2i>> filtered_lines;
	splitLinear(lines, filtered_lines);
	std::cout << "number of lines = " << filtered_lines.size() << std::endl;
	cv::Mat original_image = cv::imread("test.jpg");
	scale = 0.25f;
	cv::resize(original_image, original_image, cv::Size(static_cast<int>(original_image.cols * scale), static_cast<int>(original_image.rows * scale)), 0, 0, cv::INTER_AREA);
	for each(auto& line in filtered_lines)
	{
		if(line.size() <= 0)
			std::cout << "fuck" << std::endl;
		cv::line(original_image, line.front(), line.back(), cv::Scalar(0, 0, 255), 1);
		//for each(auto& point in line)
		//{
		//	if(point.x < 0 || original_image.cols <= point.x || point.y < 0 || original_image.rows <= point.y)
		//	{
		//		std::cout << point.x << " " << point.y << " --- " << original_image.cols << " " << original_image.rows << std::endl;
		//		continue;
		//	}
		//	original_image.at<cv::Vec3b>(point.y, point.x) = cv::Vec3b(0, 0, 255);
		//}
	}
	std::cout << "total = " << float(std::clock() - begin_time) / static_cast<float>(CLOCKS_PER_SEC) << std::endl;
	cv::imwrite("result.png", original_image);

	return 0;
}
