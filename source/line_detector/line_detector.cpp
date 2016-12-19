#include "stdafx.h"
#include "line_detector.h"
#include <functional>
#include <array>
#include <iostream>

LineDetector::LineDetector() :
	m_kernel_size(3),
	m_kernels()
{
	m_kernels.push_back((cv::Mat_<float>(m_kernel_size, m_kernel_size) <<
		0, 0, -3,
		1, 4,  0,
		0, 1,  0));
	m_kernels.push_back((cv::Mat_<float>(m_kernel_size, m_kernel_size) <<
		-3, 0, 0,
		 0, 4, 1,
		 0, 1, 0));
	m_kernels.push_back((cv::Mat_<float>(m_kernel_size, m_kernel_size) <<
		0, 1,  0,
		1, 4,  0,
		0, 0, -3));
	m_kernels.push_back((cv::Mat_<float>(m_kernel_size, m_kernel_size) <<
		 0, 1, 0,
		 0, 4, 1,
		-3, 0, 0));
	m_kernels.push_back((cv::Mat_<float>(m_kernel_size, m_kernel_size) <<
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1));
	m_kernels.push_back((cv::Mat_<float>(m_kernel_size, m_kernel_size) <<
		-1, 0, -1,
		 0, 5,  0,
		-1, 0, -1));
}

void LineDetector::detect(const cv::Mat& image)
{
	//skeleton is an image of the same size as 'image'
	//white lines(tracks) of 1-pixel with, 8-connected, on black background
	cv::Mat skeleton;
	getSkeleton(image, skeleton);
	////////////////////////////////////////////////////////////////
	//skeleton vectorization
	//in tracks we have pixel-by-pixel records of each track in skeleton
	//each track has only two ends and no junctions
	//each junction in skeleton is treated as endpoint
	//btw, we will use this information about junctions later in 'unite_tracks'
	std::vector<std::vector<cv::Point2i>> tracks;
	trackSkeleton(skeleton, tracks);
	////////////////////////////////////////////////////////////////
	//unite broken tracks
	//first we said that all junctions are endpoints, but in fact they are not
	//for each junction point we now decide, which pair of broken segments to unite
//    tracks = filter(lambda x: x is not None, unite_tracks(tracks))
/*
    tracks = filter(lambda x: x is not None, unite_tracks(tracks))
    ################################################################
    # some tracks are false positives due to very sensitive edge detection
    # here we confirm true positives (where there is a real line)
    # and filter false positives (no real line)
    tracks = confirm_tracks(tracks, img)
    ################################################################
    # here we find linear (or, almost linear) parts in each track
    # sort of piecewise linear approximation
    lines = get_lines(tracks)
    ################################################################
    # some extracted lines are really parts of one line, broken by some reason
    # here we try to unite at least some of such broken lines, in obvious cases
    lines = afterall_unite_lines(lines)
    return lines
*/
}

void LineDetector::getSkeleton(const cv::Mat& image, cv::Mat& edges) const
{
	//enhance contrast
	cv::Mat equalized_image;
	equalizeHistograms(image, equalized_image);

	//adaptive manifold filter
	//edge aware filter: remove noise, sharpen edges
	cv::Mat filtered_image;
	cv::ximgproc::amFilter(equalized_image, equalized_image, filtered_image, 3, 0.5, true);

	//canny
	cv::blur(filtered_image, filtered_image, cv::Size(3, 3));
	cv::Canny(filtered_image, edges, 200, 300, 5, true);

	//canny returns 4-connected lines, we need 8-connected
	skeletonize(edges);
}

void LineDetector::equalizeHistograms(const cv::Mat& image, cv::Mat& equalized_image) const
{
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	for each(auto& channel in channels)
	{
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
		clahe->apply(channel, channel);
	}
	cv::merge(channels, equalized_image);
}

void LineDetector::skeletonize(cv::Mat& edges) const
{
	//with k1-k4 we transform situations like:
	//
	//0 0 0		but not		0 0 X
	//X X 0					X X 0
	//0 X 0					0 X 0
	//
	//into
	//
	//0 0 0
	//X 0 0
	//0 X 0
	//
	//we have to do that for each kernel sequentially,
	//because each pass simplifies task for following passes, and we loose less information
	{
		FilterPredicate predicate = [](const cv::Mat& image, const int row, const int column) -> bool {
			return (image.at<float>(row, column) == 6.0f);
		};
		for(size_t kernel_counter = 0; kernel_counter < m_kernels.size(); ++kernel_counter)
			applyFilter(m_kernels[kernel_counter], edges, predicate);
	}

	//here we just remove very complicated situations,
	//where central pixel is surrounded by 4 or more white pixels
	//(excluding diagonal cross situation:
	//X 0 X
	//0 X 0
	//X 0 X
	//we might deal with this situation)
	//
	//of course we loose some connections,
	//but in most cases very compilcated connections are useless
	FilterPredicate predicate = [](const cv::Mat& image, const int row, const int column) -> bool {
		return (0 < image.at<float>(row, column) && image.at<float>(row, column) <= 4.0f);
	};
	applyFilter(m_kernels[4], edges, predicate);

	cv::Mat float_edges;
	edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);

	cv::Mat res1;
	cv::filter2D(float_edges, res1, -1, m_kernels[4]);
	cv::Mat res2;
	cv::filter2D(float_edges, res2, -1, m_kernels[5]);

	for(int row = 0; row < edges.rows; ++row)
		for(int column = 0; column < edges.cols; ++column)
			if(0 < res1.at<float>(row, column) && res1.at<float>(row, column) <= 5 && res2.at<float>(row, column) != 1)
				edges.at<unsigned char>(row, column) = 0;
}

void LineDetector::applyFilter(const cv::Mat kernel, cv::Mat& edges, FilterPredicate predicate) const
{
	cv::Mat float_edges;
	edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);
	cv::filter2D(float_edges, float_edges, -1, kernel);

	for(int row = 0; row < edges.rows; ++row)
	{
		for(int column = 0; column < edges.cols; ++column)
			if(predicate(float_edges, row, column))
				edges.at<unsigned char>(row, column) = 0;
	}
}

void LineDetector::trackSkeleton(cv::Mat& edges, std::vector<std::vector<cv::Point2i>>& lines) const
{
	//first we make black 1-width borders
	//almost same as if we expand original image, but saves cpu usage
	setBorderToZero(edges);

	//find endpoints and junctions
	//endpoints are stored in 'eps'
	//and junctions are marked as 127 inplace in 'skel'
	std::vector<cv::Point2i> endpoints;
	junctionAndEndpoint(edges, endpoints);

	//follow and record tracks from each active endpoint
	//when a track stops, we disable the endpoint, where it stopped
	lines.clear();
	for each(auto& point in endpoints)
	{
		//check if this endpoint is enabled in 'skel'
		if(edges.at<unsigned char>(point.y, point.x) == 255)
		{
			std::vector<cv::Point2i> line;
			trackLine(edges, point, line);
			lines.push_back(line);
		}
	}

	//juncpoints - list of junction points, they were marked as '127' in 'skel' before
	std::vector<cv::Point2i> juncpoints;
	for(int row = 0; row < edges.rows; ++row)
	{
		for(int column = 0; column < edges.cols; ++column)
		{
			if(edges.at<unsigned char>(row, column) == 127)
				juncpoints.push_back(cv::Point2i(column, row));
		}
	}
	auto lambda_less = [](const cv::Point2i& p1, const cv::Point2i& p2) -> bool
	{
		if(p1.x < p2.x)
			return true;
		return p1.y < p2.y;
	};
	std::sort(juncpoints.begin(), juncpoints.end(), lambda_less);

	//we must use junction points as endpoints to track lines,
	//because a track can start and end with junction
	for(int epoch = 0; epoch < 4; epoch++)
	{
		for each(auto juncpoint in juncpoints)
		{
			int sum = 0;
			for(int row = -1; row < 2; ++row)
			{
				for(int column = -1; column < 2; ++column)
				{
					if(juncpoint.y + row < 0 || edges.rows <= juncpoint.y + row || juncpoint.x + column < 0 || edges.cols <= juncpoint.x + column)
						continue;
					sum += edges.at<unsigned char>(juncpoint.y + row, juncpoint.x + column);
				}
			}
			if(sum - edges.at<unsigned char>(juncpoint.y, juncpoint.x) > 0)
			{
				edges.at<unsigned char>(juncpoint.y, juncpoint.x) = 255;
				std::vector<cv::Point2i> line;
				trackLine(edges, juncpoint, line);
				lines.push_back(line);
			}
		}
	}

	//after all endpoints and junction points are tracked and disabled,
	//there still could be white points at 'skel' - closed loops with no ends
	//for each loop we break it in some place and then track it

	//first remove junction points(127-s), to leave only 255-s
	cv::threshold(edges, edges, 254, 255, cv::THRESH_BINARY);

	//and then find external contours
	//with "approx_tc89_kcos" and approxPolyDP we approximate contour and get only key points,
	//not to break this loop in its linear part
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);

	endpoints.clear();
	for each(auto& contour in contours)
	{
		cv::approxPolyDP(contour, contour, 2.0f, true);
		cv::Point2i point = contour[0];
		//search for white pixels near contour point
		//we search in 5x5 square, but not in 3x3, because contour approximation can shift contour
		//a bit from real pixels
		cv::Point2i shift_point;
		bool done = false;
		for(int column = -2; column <= 3; ++column)
		{
			for(int row = -2; row <= 3; ++row)
			{
				if(edges.at<unsigned char>(point.y + row, point.x + column) > 0)
				{
					shift_point.x = column;
					shift_point.y = row;
					done = true;
				}
				if(done)
					break;
			}
			if(done)
				break;
		}
		//y_,x_ = np.transpose(np.nonzero(skel[y-2:y+3, x-2:x+3]))[0]
		endpoints.push_back(point + shift_point - cv::Point2i(2, 2));
	}

	//finally track each closed loop
	for each(auto& point in endpoints)
	{
		std::vector<cv::Point2i> line;
		trackLine(edges, point, line);
		lines.push_back(line);
	}
}

void LineDetector::setBorderToZero(cv::Mat& image) const
{
	//TODO refactor this
	for(int row = 0; row < image.rows; ++row)
	{
		image.at<unsigned char>(row, 0) = 0;
		image.at<unsigned char>(row, image.cols - 1) = 0;
	}
	for(int column = 0; column < image.cols; ++column)
	{
		image.at<unsigned char>(0, column) = 0;
		image.at<unsigned char>(image.rows - 1, column) = 0;
	}
}

void LineDetector::junctionAndEndpoint(cv::Mat& edges, std::vector<cv::Point2i>& endpoints) const
{
	cv::Mat kernel = (cv::Mat_<double>(m_kernel_size, m_kernel_size) <<
		  1,   2,  4,
		128, 256,  8,
		 64,  32, 16);
	FilterPredicate predicate = [](const cv::Mat& image, const int row, const int column) -> bool {
		//single points
		return image.at<float>(row, column) == 256;
	};
	applyFilter(kernel, edges, predicate);

	{
		//values meaning junction
		std::array<float, 13> values = {
			256+128+32+4, 256+128+2+16, 256+2+8+64, 256+8+32+1,
			256+1+64+8, 256+1+4+32, 256+4+16+128, 256+16+64+2,
			256+1+4+16, 256+4+16+64, 256+16+64+1, 256+64+1+4, 256+1+4+16+64
		};
		cv::Mat float_edges;
		edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);
		for(int row = 0; row < edges.rows; ++row)
		{
			for(int column = 0; column < edges.cols; ++column)
			{
				for(size_t value_counter = 0; value_counter < values.size(); value_counter++)
				{
					if(float_edges.at<float>(row, column) == values[value_counter])
					{
						edges.at<unsigned char>(row, column) = 127;
						break;
					}
				}
			}
		}
	}

	//values for endpoints
	endpoints.clear();

	cv::Mat float_edges;
	edges.convertTo(float_edges, CV_32F, 1.0f / 255.0f);

	for(int row = 0; row < float_edges.rows; ++row)
	{
		for(int column = 0; column < float_edges.cols; ++column)
		{
			const std::array<float, 8> values = {288, 384, 258, 264, 257, 272, 320, 260};
			for(size_t value_counter = 0; value_counter < values.size(); value_counter++)
			{
				if(float_edges.at<float>(row, column) == values[value_counter])
				{
					endpoints.push_back(cv::Point2i(column, row));
					break;
				}
			}
		}
	}

	std::sort(endpoints.begin(), endpoints.end(), [](const cv::Point2i& p1, const cv::Point2i& p2) -> bool
	{
		if(p1.x < p2.x)
			return true;
		return p1.y < p2.y;
	});
}

void LineDetector::trackLine(cv::Mat& edges, const cv::Point2i& point, std::vector<cv::Point2i>& line) const
{
	line.clear();
	line.push_back(point);

	cv::Point2i next_point;
	cv::Point2i current_point = point;
	while(nextPoint(edges, current_point, next_point))
	{
		//next_point returns white pixel position, neighbouring to current position
		//and disables current pixel in 'skel'
		//stop, if no next white pixels in 3x3 field
		current_point = next_point;
		line.push_back(current_point);
	}
}

bool LineDetector::nextPoint(cv::Mat& edges, const cv::Point2i& current_point, cv::Point2i& next_point) const
{
	std::vector<cv::Point2i> potential_points;

	//if current pixel is regular point (white)
	if(edges.at<unsigned char>(current_point.y, current_point.x) == 255)
	{
		//disable it
		edges.at<unsigned char>(current_point.y, current_point.x) = 0;
		//and search for non-black pixels in 3x3 field
		SearchPredicate predicate = [](const cv::Mat& image, const cv::Point2i& point) -> bool { return (image.at<unsigned char>(point.y, point.x) != 0); };
		getPotentialPoints(edges, current_point, potential_points, predicate);

		if(potential_points.size() <= 0)
			return false;

		if(potential_points.size() == 1)
		{
			//if only one non-black, return it
			next_point = current_point + potential_points[0];
			return true;
		}

		if(potential_points.size() > 1)
		{
			//if more than one non-black,
			//check junction points(127) and return first junction point
			SearchPredicate pred = [](const cv::Mat& image, const cv::Point2i& point) -> bool { return (image.at<unsigned char>(point.y, point.x) == 127); };
			getPotentialPoints(edges, current_point, potential_points, pred);
			if(potential_points.size() == 0)
			{
				SearchPredicate pr = [](const cv::Mat& image, const cv::Point2i& point) -> bool { return (image.at<unsigned char>(point.y, point.x) == 255); };
				getPotentialPoints(edges, current_point, potential_points, pr);
			}
			next_point = current_point + potential_points[0];
			return true;
		}
	}
	else
	{
		if(edges.at<unsigned char>(current_point.y, current_point.x) == 127)
		{
			//or, if current pixel is junction point
			//disable this pixel only if there is no other connections to this junction point
			SearchPredicate predicate = [](const cv::Mat& image, const cv::Point2i& point) -> bool { return (image.at<unsigned char>(point.y, point.x) != 255); };
			getPotentialPoints(edges, current_point, potential_points, predicate);
			if(potential_points.size() == 1)
				edges.at<unsigned char>(current_point.y, current_point.x) = 0;
		}
	}

	return false;
}

void LineDetector::getPotentialPoints(const cv::Mat& edges, const cv::Point2i& point, std::vector<cv::Point2i>& potential_points, SearchPredicate predicate) const
{
	potential_points.clear();
	for(int row = -1; row < 2; ++row)
		for(int column = -1; column < 2; ++column)
			if(predicate(edges, cv::Point2i(point.y + row, point.x + column)))
				potential_points.push_back(cv::Point2i(column, row));
}

void LineDetector::uniteTracks(const std::vector<std::vector<cv::Point2i>>& lines, std::vector<std::vector<cv::Point2i>>& united_lines) const
{
	Endpoints endpoints;
	for(size_t line_counter = 0; line_counter < lines.size(); ++line_counter)
	{
		const auto& line = lines[line_counter];
		endpoints.insert(std::make_pair(line[0], std::make_pair(line_counter, 0)));
		endpoints.insert(std::make_pair(line[line.size() - 1], std::make_pair(line_counter, 1)));
	}

	//graph is unordered multimap, graph is bidirectional
	//Key		int					- line index
	//Value		int, int, etc		- line indices
	std::unordered_multimap<int, int> graph;
	buildGraph(lines, endpoints, graph);

	std::deque<int> initial_nodes;
	getInitialNodes(graph, initial_nodes);

	std::vector<std::vector<int>> chains;
	findChains(graph, initial_nodes, chains);

	assembleLines(chains, lines, united_lines);
}

void LineDetector::buildGraph(const std::vector<std::vector<cv::Point2i>>& lines, const Endpoints& endpoints, std::unordered_multimap<int, int>& graph) const
{
	auto iter = endpoints.begin();
	while(iter != endpoints.end())
	{
		auto range = endpoints.equal_range(iter->first);
		auto const ep = range.first->first;
		const auto number_of_connections = std::distance(range.first, range.second);
		switch(number_of_connections)
		{
			case 0:
			case 1:
				break;
			case 2:
				{
					auto range_iter = range.first;
					const auto id1 = range_iter->second.first;
					++range_iter;
					const auto id2 = range_iter->second.first;
					addEdge(graph, id1, id2);
					break;
				}
			case 3:
			case 4:
				{
					auto range_iter = range.first;
					const auto id1 = range_iter->second.first;
					const auto p1 = range_iter->second.second;
					++range_iter;
					const auto id2 = range_iter->second.first;
					const auto p2 = range_iter->second.second;
					++range_iter;
					const auto id3 = range_iter->second.first;
					const auto p3 = range_iter->second.second;

					const auto& l1 = lines[id1];
					const auto& l2 = lines[id2];
					const auto& l3 = lines[id3];

					float d1;
					float d2;
					float d3;

					//get the last one from first five points
					//get the first one from last five points
					if(p1 == 0)	d1 = dist(l1[(l1.size() > 4 ? 4 : l1.size() - 1)], ep);
					else		d1 = dist(l1[(l1.size() - 5 >= 0 ? l1.size() - 5 : 0)], ep);
					if(p2== 0)	d2 = dist(l2[(l2.size() > 4 ? 4 : l2.size() - 1)], ep);
					else		d2 = dist(l2[(l2.size() - 5 >= 0 ? l2.size() - 5 : 0)], ep);
					if(p3 == 0)	d3 = dist(l3[(l3.size() > 4 ? 4 : l3.size() - 1)], ep);
					else		d3 = dist(l3[(l3.size() - 5 >= 0 ? l3.size() - 5 : 0)], ep);

					if(number_of_connections == 3)
					{
						if(d1 <= d2 && d1 <= d3)	addEdge(graph, id2, id3);
						if(d2 <= d1 && d2 <= d3)	addEdge(graph, id1, id3);
						if(d3 <= d1 && d3 <= d2)	addEdge(graph, id1, id2);
						break;
					}

					++range_iter;
					const auto id4 = range_iter->second.first;
					if(d1 <= d2 && d1 <= d3)
					{
						addEdge(graph, id2, id3);
						addEdge(graph, id1, id4);
					}
					if(d2 <= d1 && d2 <= d3)
					{
						addEdge(graph, id1, id3);
						addEdge(graph, id2, id4);
					}
					if(d3 <= d1 && d3 <= d2)
					{
						addEdge(graph, id1, id2);
						addEdge(graph, id3, id4);
					}

					break;
				}
			default:
				std::cout << "Warning! unite lines doesn't support more than 4 connections" << std::endl;
				break;
		}
		++iter;
	}
}

void LineDetector::addEdge(std::unordered_multimap<int, int>& graph, const int vertex1, const int vertex2) const
{
	auto range = graph.equal_range(vertex1);
	bool has_edge = false;
	for(auto iter = range.first; iter != range.second; ++iter)
		if(iter->second == vertex2)
		{
			has_edge = true;
			break;
		}
	if(!has_edge)
		graph.insert(std::make_pair(vertex1, vertex2));

	range = graph.equal_range(vertex2);
	has_edge = false;
	for(auto iter = range.first; iter != range.second; ++iter)
		if(iter->second == vertex1)
		{
			has_edge = true;
			break;
		}
	if(!has_edge)
		graph.insert(std::make_pair(vertex2, vertex1));
}

float LineDetector::dist(const cv::Point2i& p1, const cv::Point2i& p2) const
{
	cv::Point p = p1 - p2;
	return sqrt(static_cast<float>(p.x * p.x + p.y * p.y));
}

void LineDetector::getInitialNodes(const std::unordered_multimap<int, int>& graph, std::deque<int>& initial_nodes) const
{
	initial_nodes.clear();
	auto iter = graph.begin();
	while(iter != graph.end())
	{
		auto range = graph.equal_range(iter->first);
		//if degree of the node equal 1 than it is initial node
		if(std::distance(range.first, range.second) == 1)
			initial_nodes.push_back(range.first->first);
		iter = range.second;
	}
}

void LineDetector::findChains(std::unordered_multimap<int, int>& graph, std::deque<int>& initial_nodes, std::vector<std::vector<int>>& chains) const
{
	while(!initial_nodes.empty())
	{
		auto current_node = initial_nodes.front();
		initial_nodes.pop_front();

		std::vector<int> current_chain;
		current_chain.push_back(current_node);

		while(true)
		{
			auto range = graph.equal_range(current_node);

			//if has no edges with this node
			if(range.first == range.second)
				break;

			int next_node = range.first->second;
			current_chain.push_back(next_node);
			removeNode(graph, current_node);
			current_node = next_node;
		}

		if(current_chain.size() > 1)
			chains.push_back(current_chain);
	}
}

void LineDetector::removeNode(std::unordered_multimap<int, int>& graph, const int node) const
{
	auto iter = graph.begin();
	while(iter != graph.end())
	{
		auto range = graph.equal_range(iter->first);
		auto range_iter = range.first;
		while(range_iter != range.second)
		{
			if(range_iter->first == node || range_iter->second == node)
				range_iter = graph.erase(range_iter);
			else
				++range_iter;
		}
		iter = range.second;
	}
}

void LineDetector::assembleLines(const std::vector<std::vector<int>>& chains, const std::vector<std::vector<cv::Point2i>>& lines, std::vector<std::vector<cv::Point2i>>& united_lines) const
{
	std::vector<int> used_lines;
	used_lines.resize(lines.size(), -1);

	united_lines.clear();

	for each(auto chain in chains)
	{
		std::vector<cv::Point2i> line;
		chainToLine(chain, lines, line, used_lines);
		united_lines.push_back(line);
	}

	for(size_t line_counter = 0; line_counter < lines.size(); ++line_counter)
	{
		if(used_lines[line_counter] < 0)
			united_lines.push_back(lines[line_counter]);
	}
}

void LineDetector::chainToLine(const std::vector<int>& chain, const std::vector<std::vector<cv::Point2i>>& lines, std::vector<cv::Point2i>& line, std::vector<int>& used_lines) const
{
	line.clear();

	used_lines[chain[0]] = 1;
	line = lines[chain[0]];

	for(size_t link_counter = 1; link_counter < chain.size(); ++link_counter)
	{
		used_lines[chain[link_counter]] = 1;
		auto next_line = lines[chain[link_counter]];

		const auto p11 = line[0];
		const auto p12 = line[line.size() - 1];
		const auto p21 = next_line[0];
		const auto p22 = next_line[next_line.size() - 1];

		if(p11 == p21)
		{
			std::reverse(line.begin(), line.end());
			line.insert(line.end(), next_line.begin(), next_line.end());
			continue;
		}
		if(p12 == p21)
		{
			line.insert(line.end(), next_line.begin(), next_line.end());
			continue;
		}
		if(p11 == p22)
		{
			line.insert(line.begin(), next_line.begin(), next_line.end());
			continue;
		}
		if(p12 == p22)
		{
			line.insert(line.end(), next_line.rbegin(), next_line.rend());
		}
	}
}
