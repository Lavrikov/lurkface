///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define StateTypeK Rect_<float>


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
    KalmanTracker()
    {
        init_kf(StateTypeK());
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
    }
    KalmanTracker(StateTypeK initRect)
    {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;
    }

    ~KalmanTracker()
    {
        m_history.clear();
    }

    StateTypeK predict();
    void update(StateTypeK stateMat);
    
    StateTypeK get_state();
    StateTypeK get_rect_xysr(float cx, float cy, float s, float r);

    static int kf_count;

    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

private:
    void init_kf(StateTypeK stateMat);

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateTypeK> m_history;
};

class TrackingBox //This temporary object is used to add new bboxes detections to a trackers list
{
public:
    TrackingBox();

    TrackingBox(Rect_<float> to_box, int to_id)
    {
        frame = -1;
        id = to_id;
        box = to_box;
    }

    TrackingBox(Rect_<float> to_box, int to_id, float to_confidence)
    {
        frame = -1;
        id = to_id;
        box = to_box;
        confidence = to_confidence;
    }

    int frame;
    int id;
    Rect_<float> box;
    float confidence;
};

#endif