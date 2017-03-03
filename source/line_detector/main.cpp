#include "stdafx.h"
#include "line_detector.h"
#include <ctime>
#include <iostream>
#include <fstream>

int main()
{
	cv::Mat image = cv::imread("test.jpg");
	const double scale = 0.25f;
	cv::resize(image, image, cv::Size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale)), 0, 0, cv::INTER_AREA);

	clock_t begin = clock();
	LineDetector line_detector;
	std::vector<LineDetector::Segment> lines;
	line_detector.detect(image, lines);

	std::fstream output_file("lines.txt", std::ios::out);
	if(!output_file.is_open())
	{
		std::cout << "failed to create output file" << std::endl;
		return 0;
	}
	for(int line_counter = 0; line_counter < lines.size(); ++line_counter)
	{
		output_file << lines[line_counter].p1.x << " " << lines[line_counter].p1.y << " " << lines[line_counter].p2.x << " " << lines[line_counter].p2.y << std::endl;
	}
	output_file.close();

	clock_t end = clock();
	std::cout << (end - begin) / static_cast<float>(CLOCKS_PER_SEC) << std::endl;
	return 0;
}
