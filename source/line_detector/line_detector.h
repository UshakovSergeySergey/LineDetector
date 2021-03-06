#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <unordered_map>
#include <functional>
#include <deque>
#include <array>

namespace std
{
	template<>
	struct hash<cv::Point2i>
	{
		size_t operator()(const cv::Point2i& point) const
		{
			return std::hash<int>()(point.x) ^ std::hash<int>()(point.y);
		}
	};
}

class LineDetector
{
	public:
		typedef struct
		{
			std::array<float, 4> line_equation;
			cv::Point2i p1;
			cv::Point2i p2;
			float length;
		} Segment;

	public:
		LineDetector();
		void detect(const cv::Mat& image, std::vector<Segment>& united_lines);

	private:
		typedef std::function<bool(const cv::Mat&, const int, const int)> FilterPredicate;
		typedef std::function<bool(const cv::Mat&, const cv::Point2i&)> SearchPredicate;
		typedef std::unordered_multimap<cv::Point2i, std::pair<int, int>> Endpoints;

	private:
		//get_skeleton
		void getSkeleton(const cv::Mat& image, cv::Mat& edges) const;
		void equalizeHistograms(const cv::Mat& image, cv::Mat& equalized_image) const;
		void skeletonize(cv::Mat& edges) const;
		void applyFilter(const cv::Mat kernel, cv::Mat& edges, FilterPredicate predicate) const;

		//track_skeleton
		void trackSkeleton(cv::Mat& edges, std::vector<std::vector<cv::Point2i>>& lines) const;
		void setBorderToZero(cv::Mat& image) const;
		void junctionAndEndpoint(cv::Mat& edges, std::vector<cv::Point2i>& endpoints) const;
		void trackLine(cv::Mat& edges, const cv::Point2i& point, std::vector<cv::Point2i>& line) const;
		bool nextPoint(cv::Mat& edges, const cv::Point2i& current_point, cv::Point2i& next_point) const;
		void getPotentialPoints(const cv::Mat& edges, const cv::Point2i& point, std::vector<cv::Point2i>& potential_points, SearchPredicate predicate) const;

		//unite_tracks
		void uniteTracks(const std::vector<std::vector<cv::Point2i>>& lines, std::vector<std::vector<cv::Point2i>>& united_lines) const;
		void buildGraph(const std::vector<std::vector<cv::Point2i>>& lines, const Endpoints& endpoints, std::unordered_multimap<int, int>& graph) const;
		void getInitialNodes(const std::unordered_multimap<int, int>& graph, std::deque<int>& initial_nodes) const;
		void findChains(std::unordered_multimap<int, int>& graph, std::deque<int>& initial_nodes, std::vector<std::vector<int>>& chains) const;
		void assembleLines(const std::vector<std::vector<int>>& chains, const std::vector<std::vector<cv::Point2i>>& lines, std::vector<std::vector<cv::Point2i>>& united_lines) const;
		void addEdge(std::unordered_multimap<int, int>& graph, const int vertex1, const int vertex2) const;
		float dist(const cv::Point2i& p1, const cv::Point2i& p2) const;
		void removeNode(std::unordered_multimap<int, int>& graph, const int node) const;
		void chainToLine(const std::vector<int>& chain, const std::vector<std::vector<cv::Point2i>>& lines, std::vector<cv::Point2i>& line, std::vector<int>& used_lines) const;

		//confirm_tracks
		void confirmTracks(const std::vector<std::vector<cv::Point2i>>& tracks, const cv::Mat& image, std::vector<std::vector<cv::Point2i>>& confirmed) const;
		void imageDiff(const cv::Mat& im1, const cv::Mat& im2, cv::Mat& result) const;
		void grad(const cv::Mat& image, const int d, cv::Mat& result) const;

		//get_lines
		void getLines(const std::vector<std::vector<cv::Point2i>>& tracks, std::vector<Segment>& lines) const;
		void processTrack(const std::vector<cv::Point2i>& track, std::vector<std::vector<cv::Point2i>>& lines, std::vector<Segment>& equations) const;
		float adjustLinearity(const float linearity, const std::vector<cv::Point2i>& segment) const;
		float segmentLinearity(const std::vector<cv::Point2i>& line) const;
		void fitLine(const std::vector<cv::Point2i>& points, std::array<float, 4>& line_equation) const;
		bool isInCorridor(const cv::Point2i& point, const std::array<float, 4>& line_equation, const float corridor_width = 2.0f) const;
		void encodeSegment(const std::vector<cv::Point2i>& segment, Segment& equation) const;
		void lineEquation(const cv::Point2f& p1, const cv::Point2f& p2, std::array<float, 4>& line_equation) const;
		float distanceToLine(const cv::Point2i& point, const std::array<float, 4>& line_equation) const;
		cv::Point2i perpPoint(const std::array<float, 4>& line_equation, const cv::Point2i point) const;
		cv::Point2i intersectPoint(const float a1, const float b1, const float c1, const float a2, const float b2, const float c2) const;

		//afterall_unite_lines
		void afterallUniteLines(const std::vector<Segment>& lines, std::vector<Segment>& united_lines) const;
		void computeAngles(const std::vector<Segment>& lines, std::vector<std::pair<float, int>>& angles) const;
		void buildGraph(const std::vector<Segment>& lines, const std::vector<std::pair<float, int>>& angles, std::unordered_multimap<int, int>& graph) const;
		void assembleLines(const std::vector<std::vector<int>>& chains, const std::vector<Segment>& lines, std::vector<Segment>& united_lines) const;
		void chainToLine(const std::vector<int>& chain, const std::vector<Segment>& lines, Segment& line, std::vector<int>& used_lines) const;

		bool isInside(const cv::Mat& image, const cv::Point2i& point) const
		{
			return (0 <= point.x && point.x < image.cols && 0 <= point.y && point.y < image.rows);
		}

	private:
		const int m_kernel_size;
		std::vector<cv::Mat> m_kernels;
};
