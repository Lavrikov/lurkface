#include "LurkLib.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>

//sort dependences
#include "sort_source\Hungarian.h"
#include "sort_source\KalmanTracker.h"
#include "opencv2/video/tracking.hpp"

//caffe dependences
#include "caffe\caffe.hpp"
#include "opencv2\opencv.hpp"
#include <string>
#include <vector>

#ifndef NUMBER_OF_RESULTS_IN_SSD_VECTOR
#define NUMBER_OF_RESULTS_IN_SSD_VECTOR 7
#endif //NUMBER_OF_RESULTS_IN_SSD_VECTOR

/* For each detected object detection format: [image_id, label, score, xmin, ymin, xmax, ymax].*/
typedef std::vector<std::vector<float> > Prediction;

using namespace cv;
using namespace caffe;  // NOLINT(build/namespaces)


//////////////////////////////////////////////////////////////////////////////////
// Detection Engine definition
class DetectorSSD 
{
public:
	DetectorSSD();
	DetectorSSD(const wchar_t * pathToDeployFile, const wchar_t * pathToCaffeModelFile,
		const float trek_trsh, const std::string &mean_value) :
		m_trekTrsh(trek_trsh) //initial values for const parameters
	{
		// Set calculating device GPU-CPU
		Caffe::set_mode(Caffe::GPU);

		/* Load the network. */
		try
		{
			net_.reset(new Net<float>(pathToDeployFile, TEST));//load layers map
			net_->CopyTrainedLayersFrom(pathToCaffeModelFile);//load weights
		}
		catch (const std::runtime_error& e)
		{
			std::cout << "CaffeInitializationError" << std::endl;
		}

		Blob<float>* input_layer = net_->input_blobs()[0];
		num_channels_ = input_layer->channels();
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

		if (!mean_value.empty())
		{
			SetMean(mean_value);//set well established mean values "123,104,117" for every color B,G,R, you should change this parameter if you used other values during training
		}
	}
	void Detect(const char* uuid, const int* widths, const int* heights, const int* strides, const unsigned char* frameDataRGB, int* nResults, float** results);

		~DetectorSSD(){};
private:

	void setPipeline(std::vector<cv::Mat>* input_channels);

    void SetMean(const string& mean_value);

    void dataToBlob(const cv::Mat& img, std::vector<cv::Mat>* input_channels, double normalize_value);

    double GetIOU(const Rect_<float>& bb_test, const Rect_<float>& bb_gt);// Computes IOU between two bounding boxes

    void Sort(const vector<TrackingBox>& detData, Prediction& outputs);//Post-processing of SSD output bboxes to avoid missing faces between video frames


private:
    int m_frameWidth, m_frameHeight, m_frameStride; 
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::shared_ptr<Net<float> > net_;//deploy network
    float m_nor_val = 1.0;
    bool m_useMeanFile;
	//The confidence threshold for creating new trackers. Existing trackers living until they confirmed with suitable detections.
    //regular values 0.1-0.3. particular threshold is selecting with video.
	//The threshold should be lower than maximal observing face confidence and higher than normal background confidence
	const float m_trekTrsh;

    //////////////////////////////////tracking 
    vector<KalmanTracker> m_trackers;// Trackers for the every object recognized as face with high confidence at least once
    float m_sort_Scale = 100; //the scale factor is using for converting relative coordinates (SSD makes predictions 0..1, Tracker requires 0..some value, pixels, it can be any value because we recalculate relative coordinates for Tracker output 0..1)
    const float shift_left = 0.15;//bbox scale (An experimental value, a low value tends to some faces missing by tracker,a high value leads to cover to much area around a face)
    const float shift_right = 0.25;//bbox scale ----
};

