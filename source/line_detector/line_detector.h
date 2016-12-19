#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <unordered_map>
#include <functional>
#include <deque>

class LineDetector
{
	public:
		LineDetector();
		void detect(const cv::Mat& image);

	private:
		static size_t pointHash(const cv::Point2i& point);

		typedef std::function<bool(const cv::Mat&, const int, const int)> FilterPredicate;
		typedef std::function<bool(const cv::Mat&, const cv::Point2i&)> SearchPredicate;
		typedef std::unordered_multimap<cv::Point2i, std::pair<int, int>, std::function<decltype(pointHash)>> Endpoints;

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

	private:
		const int m_kernel_size;
		std::vector<cv::Mat> m_kernels;
};
