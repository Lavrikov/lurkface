#include "Detector.h"



// Computes IOU between two bounding boxes ( SORT)
double DetectorSSD::GetIOU(const Rect_<float>& bb_test,const Rect_<float>& bb_gt)
{
    const float in = (bb_test & bb_gt).area();
    const float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

void DetectorSSD::Sort(const vector<TrackingBox>& detData, Prediction& outputs)
{
    //Source  https://github.com/mcximing/sort-cpp
    //This is a C++ implementation of SORT, a simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences. Original Python code and publication information found at https ://github.com/abewley/sort, By Alex Bewley
    // SORT hyper parameters
    const int max_age = 3; //lifetime of each tracker without confirming from detected bboxes. This value sets number of video frames within ones SORT draw bbox predictions with known speed and scale and without detections form SSD. 
    const int min_hits = 1; //helps count dead trackers
    const double iouThreshold = 0.3; // threshold of differences between two bboxes(from two frames) to mark them as one object.
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

    
    // variables used in the for-loop
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;

    // We skip SORT body in case of detections lack (otherwise SORT crashes)
    if (m_trackers.empty())
    {
        // initialize kalman m_trackers using first detections.
        for (unsigned int i = 0; i < detData.size(); ++i)
        {
            // Here we add a new tracker for face at least once detected with a high confidence
            if (detData[i].confidence > m_trekTrsh)
            {
                KalmanTracker trk = KalmanTracker(detData[i].box);
                m_trackers.push_back(trk);
            }
        }
    }
    else
    {
        ///////////////////////////////////////tracker body
        // 1. get predicted locations from existing m_trackers.
        predictedBoxes.clear();
        for (auto it = m_trackers.begin(); it != m_trackers.end();)
        {
            const Rect_<float> pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                ++it;
            }
            else
            {
                it = m_trackers.erase(it);
            }
        }

        ///////////////////////////////////////
        // 2. associate detections to tracked object (both represented as bounding boxes)
        const unsigned int trkNum = predictedBoxes.size();
        // Here we skip this part of SORT in case of tracker predictions lack (otherwise, happens a rare bug for range of resolutions in case of the only tracker going to minus coordinates)
        if (trkNum > 0)
        {            
            const unsigned int detNum = detData.size();
            iouMatrix.clear();
            iouMatrix.resize(trkNum, vector<double>(detNum, 0));

            for (unsigned int i = 0; i < trkNum; ++i) // compute iou matrix as a distance matrix
                for (unsigned int j = 0; j < detNum; ++j)
                    iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
 
            // solve the assignment problem using hungarian algorithm.
            // the resulting assignment is [track(prediction) : detection], with len=preNum
            HungarianAlgorithm HungAlgo;
            assignment.clear();
            HungAlgo.Solve(iouMatrix, assignment);
            // find matches, unmatched_detections and unmatched_predictions
            unmatchedTrajectories.clear();
            unmatchedDetections.clear();
            allItems.clear();
            matchedItems.clear();
            if (detNum > trkNum) //	there are unmatched detections
            {
                for (unsigned int n = 0; n < detNum; ++n)
                    allItems.insert(n);

                for (unsigned int i = 0; i < trkNum; ++i)
                    matchedItems.insert(assignment[i]);

                set_difference(allItems.begin(), allItems.end(),
                    matchedItems.begin(), matchedItems.end(),
                    insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
            }
            else if (detNum < trkNum) // there are unmatched trajectory/predictions
            {
                for (unsigned int i = 0; i < trkNum; ++i)
                    if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                        unmatchedTrajectories.insert(i);
            }
            // filter out matched with low IOU
            matchedPairs.clear();
            for (unsigned int i = 0; i < trkNum; ++i)
            {
                if (assignment[i] == -1) // pass over invalid values
                    continue;
                if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
                {
                    unmatchedTrajectories.insert(i);
                    unmatchedDetections.insert(assignment[i]);
                }
                else
                {
                    matchedPairs.push_back(cv::Point(i, assignment[i]));
                }
            }

            ///////////////////////////////////////
            // 3. updating m_trackers
            // update matched m_trackers with assigned detections.
            // each prediction is corresponding to a tracker
            int detIdx, trkIdx;
            for (unsigned int i = 0; i < matchedPairs.size(); ++i)
            {
                trkIdx = matchedPairs[i].x;
                detIdx = matchedPairs[i].y;
                m_trackers[trkIdx].update(detData[detIdx].box);
            }

            // create and initialise new m_trackers for unmatched detections
            for (auto umd : unmatchedDetections)
            {
                //Here we add to tracker high confidence detected faces only. Further, the tracker algorithm searches the 
                //same faces among detections with low confidence scores. Experimentally, such confidence is often  
                //hesitating about threshold level. Especially, for fast detection algorithms.   
                if (detData[umd].confidence > m_trekTrsh)
                {
                    KalmanTracker tracker = KalmanTracker(detData[umd].box);
                    m_trackers.push_back(tracker);
                }
            }
            // get m_trackers' output
            frameTrackingResult.clear();
            for (auto it = m_trackers.begin(); it != m_trackers.end();)
            {
               
                if (((*it).m_time_since_update < 1) &&
                    ((*it).m_hit_streak >= min_hits))
                {
                    frameTrackingResult.push_back(TrackingBox((*it).get_state(),(*it).m_id + 1));
                    ++it;
                }
                else
                {
                    ++it;
                }

                // remove dead tracklet// don't move otherwise appears void vectors, but SORT should has at least one tracker to work correctly
                if (it != m_trackers.end() && (*it).m_time_since_update > max_age)
                    it = m_trackers.erase(it);
            }

            /* Copy the output of SORT to a std::vector */
            // Here we send to output SORT confirmed detections bboxes only
            for (auto tb : frameTrackingResult)
            {
                vector<float> detection(NUMBER_OF_RESULTS_IN_SSD_VECTOR);
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                detection = {0, 0, tb.confidence, tb.box.x / m_sort_Scale, tb.box.y / m_sort_Scale, (tb.box.x + tb.box.width) / m_sort_Scale, (tb.box.y + tb.box.height) / m_sort_Scale};
				outputs.push_back(detection);
            }
        }//skip tracker running in case of detections lack
    }//skip SORT filters in case of existing trackers lack
}


void DetectorSSD::Detect(const char* uuid, const int* widths, const int* heights, const int* strides, const unsigned char* frameDataRGB, int* nResults, float** results)
{
	cv::Mat frame;
	frame = cv::Mat(*heights, *widths, CV_8UC3, (void*)frameDataRGB, *strides);
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
        input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

	//put the image to the blob
    std::vector<cv::Mat> input_channels;
    setPipeline(&input_channels);
    dataToBlob(frame, &input_channels, m_nor_val);
    net_->Forward();

    //detector save data for the tracker
    vector<TrackingBox> detData;


    /* Copy the output layer to a TrackingBox */
    Blob<float>* result_blob = net_->output_blobs()[0];
    for (int i = 0; i < result_blob->num(); ++i)
    {
        float* result = result_blob->mutable_cpu_data() + i * result_blob->channels();
        const int num_det = result_blob->height();
        for (int k = 0; k < num_det; ++k)
        {

            //Here we increase bbox size to better covering faces and tracking quality
            //and automatically checking boundings of frame through selecting max and min
			//coordinates here is relative from 0 to 1.
            const float xmin = std::max(0.0f, result[3] - shift_left*(result[5] - result[3]));
            const float ymin = std::max(0.0f, result[4] - shift_left*(result[6] - result[4]));
            const float xmax = std::min(1.0f,result[5] + shift_right*(result[5] - result[3]));
            const float ymax = std::min(1.0f,result[6] + shift_right*(result[6] - result[4]));

            if (result[0] == -1 || result[1] == -1) {
                // Skip invalid detection.
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                result += NUMBER_OF_RESULTS_IN_SSD_VECTOR;
                continue;
            }

            //convert bbox to the tracker format
			if (xmin>0 && ymin>0 && xmax > 0 && ymax > 0) //trecking algorithm can handle positive coordinates of bboxes only
				//coordinates values must be in pixels format, that is why we multiply them with m_sort_Scale 
				// 0 - i dont know why?
				//also we send confidence to tracikg algoritm (result[2])
                detData.push_back(TrackingBox(Rect_<float>(Point_<float>(xmin, ymin) * m_sort_Scale, Point_<float>(xmax, ymax) * m_sort_Scale), 0, result[2]));
            
            result += NUMBER_OF_RESULTS_IN_SSD_VECTOR;
        }

    }
	Prediction output;
    //Here we select appropriate bboxes with SORT
	//The SORT method filling output with correct bboxes.
    DetectorSSD::Sort(detData, output);
	//Here we send number of detected bboxes 
	nResults[0] = output.size();
	//Here we send output to results
	for (int i = 0; i < output.size(); ++i)
	{
		for (int j = 0; j < output[i].size(); ++j)
		{
			results[i][j] = output[i][j];
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*This is custom approach to put data into the input blob of caffe.
  The cv::split in DetectorSSD::dataToBlob operation will write the separate
  channels directly to the input layer. */
void DetectorSSD::setPipeline(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    int channels = input_layer->channels();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < channels; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
		//Now, a cv::Mat channel is referring at the same memory as a caffe::input_blob.
		//So, writing any data to channel sends such data automatically to caffe:: blob
        input_channels->push_back(channel);
        input_data += width * height;
    }

}


void DetectorSSD::dataToBlob(const cv::Mat& img, std::vector<cv::Mat>* input_channels, double normalize_value) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample=img;
    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
		//Experimentally, INTER_AREA helps to achieve better detection results for downscaled images
        cv::resize(sample, sample_resized, input_geometry_,0,0,INTER_AREA);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
    cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
}

/* Set mean for image processing,  */
void DetectorSSD::SetMean(const string& mean_value) {
    cv::Scalar channel_mean;
    if (!mean_value.empty()) {
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        if (!(values.size() == 1 || values.size() == num_channels_))
        {
            std::cout<< "Specify either 1 mean_value or as many as channels: " + std::to_string(num_channels_) << std::endl;
        }
        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
                cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

// Example of exported API using
int main(int argc, char** argv)
{

	cv::VideoCapture cap("exvideo.mp4");
	if (!cap.isOpened())
	{
		std::cout << "Could not open the video!" << std::endl;
		return -1;
	}

	int totalFrameNum = cap.get(CV_CAP_PROP_FRAME_COUNT);
	const int expectedResNum = 100;//max number of produced bbox
	void* handle = 0;
	cv::Mat frame;
	cap >> frame;
	
	handle = lurkCreateDetector(L"deployFaceBoxesSfzZhang15-2.prototxt",
		L"faceboxes_original.caffemodel",0.2);
	
	float **results = nullptr;
	int* nRes;

	for (int frameNum = 0; frameNum < totalFrameNum - 10;)
	{
		cap >> frame;
		//store actual frame size (if resolution can changes)
		int* width= new int(frame.cols);
		int* height = new int(frame.rows);
		int* stride = new int(frame.step);
        int nRes[1];
        nRes[0] = 0;
		results = new float*[expectedResNum];

		//Here we create a two dimensional array for results
		for (int j = 0; j < expectedResNum; ++j)
			// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			results[j] = new float[7];
		//Here we call method processing our frame(image) with created engine 
		lurkProcessFrame(handle, "0", width, height, stride, (const unsigned char*)frame.data, nRes, (float** const)results);
		//Here we draw number of detected faces but not bigger expectedResNum
		for (int jj = 0; jj < std::min(nRes[0], expectedResNum); ++jj)
		{
            if (results[jj][0] != -1)
            {
                cv::Point pt1, pt2;
                pt1.x = (frame.cols*results[jj][3]) - 8;
                pt1.y = (frame.rows*results[jj][4]) - 8;

                if (pt1.x < 0){ pt1.x = 0; }
                if (pt1.y < 0){ pt1.y = 0; }
                pt2.x = (frame.cols*results[jj][5]);
                pt2.y = (frame.rows*results[jj][6]);
                if ((pt2.x - pt1.x)>0 && (pt2.y - pt1.y)>0){
                    Mat l_frame(frame, Rect(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y));
                    cv::resize(l_frame, l_frame, Size(), 0.2, 0.2);
                    cv::resize(l_frame, l_frame, Size(pt2.x - pt1.x, pt2.y - pt1.y));
                    l_frame.copyTo(frame.rowRange(pt1.y, pt2.y).colRange(pt1.x, pt2.x));
                }
			}
		}

		delete width;
		delete height;
		delete stride;

		for (int j = 0; j < expectedResNum; ++j)
			// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			delete[] results[j];
		if (results) delete[] results;

		results = nullptr;

		cv::imshow("frame", frame);
		cv::waitKey(1);
	}

	lurkDestroy(handle);
	return 0;
}