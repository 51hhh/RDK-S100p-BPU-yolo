#include "trajectory_utils.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {
template<int N> cv::Matx<float,N,N> inverse(const cv::Matx<float,N,N>& a) {
    return a.inv(cv::DECOMP_SVD);
}
float norm3(const cv::Point3f& p) { return std::sqrt(p.dot(p)); }
cv::Point3f point(const cv::Matx<float,6,1>& x, int o) { return {x(o),x(o+1),x(o+2)}; }
}

const char* observationSourceName(ObservationSource s) {
    switch (s) { case ObservationSource::Rgbd:return "RGB-D"; case ObservationSource::KalmanPrediction:return "PRED"; case ObservationSource::Monocular:return "MONO"; default:return "NONE"; }
}

BallTrajectoryTracker::BallTrajectoryTracker(TrackingConfig a, TrajectoryConfig b,
 LandingGateConfig c, cv::Vec3f n, float d):tracking_(a),trajectory_(b),gate_(c),ground_normal_(n),ground_offset_m_(d) {
    const float l=cv::norm(n); if(l>1e-6f) ground_normal_*=1.0f/l;
}
void BallTrajectoryTracker::reset(){initialized_=false; track_frames_=lost_frames_=stable_frames_=0; last_time_s_=0; p9_=cv::Matx<float,9,9>::eye(); p6_=cv::Matx<float,6,6>::eye();}

BallTrackResult BallTrajectoryTracker::update(const Detection* det,const DistanceMeasurement* dm,const CameraModel& cam,double now) {
    BallTrackResult out; float dt=last_time_s_>0?std::clamp(float(now-last_time_s_),0.005f,0.1f):1.0f/60; last_time_s_=now;
    bool rgbd=det&&dm&&det->score>=tracking_.min_detection_confidence&&dm->depth_valid&&dm->valid_samples>=tracking_.min_depth_samples&&dm->center_spread_m<=tracking_.max_depth_spread_m;
    if(rgbd&&dm->mono_valid&&std::abs(dm->center_xyz_m.z-dm->mono_center_z_m)>tracking_.max_rgbd_mono_delta_m) rgbd=false;
    bool mono=det&&dm&&cam.valid()&&dm->mono_valid;
    cv::Point3f z; ObservationSource src=ObservationSource::None;
    if(rgbd){z=dm->center_xyz_m;src=ObservationSource::Rgbd;}
    else if(initialized_&&lost_frames_<3){src=ObservationSource::KalmanPrediction;}
    else if(mono){float u=(det->bbox[0]+det->bbox[2])*.5f,v=(det->bbox[1]+det->bbox[3])*.5f,Z=dm->mono_center_z_m;auto q=SphereDistanceEstimator::undistortPixel({u,v},cam);z={(q.x-cam.cx)*Z/cam.fx,(q.y-cam.cy)*Z/cam.fy,Z};src=ObservationSource::Monocular;}

    cv::Matx<float,9,9> F=cv::Matx<float,9,9>::eye(); for(int i=0;i<3;i++){F(i,i+3)=dt;F(i,i+6)=.5f*dt*dt;F(i+3,i+6)=dt;}
    if(initialized_){x9_=F*x9_;p9_=F*p9_*F.t();float q=tracking_.process_accel_mps2*tracking_.process_accel_mps2*dt;for(int i=0;i<9;i++)p9_(i,i)+=q*(i<3?dt*dt:dt);}
    if(src==ObservationSource::Rgbd||src==ObservationSource::Monocular){
        if(!initialized_){x9_(0)=z.x;x9_(1)=z.y;x9_(2)=z.z;initialized_=true;track_frames_=1;p9_*=0.5f;}
        else {cv::Vec3f y(z.x-x9_(0),z.y-x9_(1),z.z-x9_(2));float rv=src==ObservationSource::Rgbd?std::max(.0025f,dm->center_spread_m*dm->center_spread_m):.25f;cv::Matx33f S(rv,0,0,0,rv,0,0,0,rv);for(int i=0;i<3;i++)for(int j=0;j<3;j++)S(i,j)+=p9_(i,j);float d2=y.dot(inverse<3>(S)*y);if(d2<tracking_.innovation_gate_chi2){auto K=p9_.get_minor<9,3>(0,0)*inverse<3>(S);x9_+=K*y;cv::Matx<float,3,9> H=cv::Matx<float,3,9>::zeros();H(0,0)=H(1,1)=H(2,2)=1.0f;p9_=(cv::Matx<float,9,9>::eye()-K*H)*p9_;track_frames_++;lost_frames_=0;}else lost_frames_++;}
    } else if(initialized_) lost_frames_++;
    if(lost_frames_>tracking_.max_lost_frames){reset();return out;} if(!initialized_)return out;

    // Layer 2: robust Student-t correction; raw RGB-D/mono only, layer-1 state is initialization/fallback.
    if(track_frames_==1){for(int i=0;i<6;i++)x6_(i)=x9_(i);p6_=cv::Matx<float,6,6>::eye();}
    cv::Point3f vel{x6_(3),x6_(4),x6_(5)};float k=.5f*trajectory_.air_density*trajectory_.drag_coefficient*float(M_PI)*trajectory_.ball_radius_m*trajectory_.ball_radius_m/trajectory_.ball_mass_kg;
    float speed=norm3(vel);cv::Point3f acc=-k*speed*vel;acc.y+=trajectory_.gravity_mps2;
    for(int i=0;i<3;i++){x6_(i)+=x6_(i+3)*dt+.5f*acc.dot(cv::Point3f(i==0,i==1,i==2))*dt*dt;x6_(i+3)+=acc.dot(cv::Point3f(i==0,i==1,i==2))*dt;}
    for(int i=0;i<6;i++)p6_(i,i)+=(i<3?trajectory_.q_position:trajectory_.q_velocity)*dt;
    float weight=1;
    if(src==ObservationSource::Rgbd||src==ObservationSource::Monocular){cv::Vec3f y(z.x-x6_(0),z.y-x6_(1),z.z-x6_(2));float rv=src==ObservationSource::Rgbd?std::max(.0025f,dm->center_spread_m*dm->center_spread_m):.25f;cv::Matx33f S(rv,0,0,0,rv,0,0,0,rv);for(int i=0;i<3;i++)for(int j=0;j<3;j++)S(i,j)+=p6_(i,j);float mahal=y.dot(inverse<3>(S)*y);weight=(trajectory_.student_t_nu+3)/(trajectory_.student_t_nu+mahal);S(0,0)+=rv/weight;S(1,1)+=rv/weight;S(2,2)+=rv/weight;auto K=p6_.get_minor<6,3>(0,0)*inverse<3>(S);x6_+=K*y;}
    out.track_valid=track_frames_>=tracking_.min_init_frames;out.position_m=point(x6_,0);out.velocity_mps=point(x6_,3);out.source=src;out.student_weight=weight;out.confidence=det?det->score:0;

    cv::Point3f p=out.position_m,v=out.velocity_mps;float t=0,step=trajectory_.rk4_dt_s,prev=ground_normal_.dot(cv::Vec3f(p.x,p.y,p.z))+ground_offset_m_;while(t<trajectory_.max_predict_time_s){auto deriv=[&](cv::Point3f pp,cv::Point3f vv){float s=norm3(vv);return std::pair<cv::Point3f,cv::Point3f>{vv,cv::Point3f(-k*s*vv.x,trajectory_.gravity_mps2-k*s*vv.y,-k*s*vv.z)};};auto a=deriv(p,v),b=deriv(p+a.first*(step*.5f),v+a.second*(step*.5f)),c=deriv(p+b.first*(step*.5f),v+b.second*(step*.5f)),d=deriv(p+c.first*step,v+c.second*step);p+=(a.first+b.first*2+c.first*2+d.first)*(step/6);v+=(a.second+b.second*2+c.second*2+d.second)*(step/6);t+=step;float plane=ground_normal_.dot(cv::Vec3f(p.x,p.y,p.z))+ground_offset_m_;if(prev*plane<=0&&t>.02f){out.landing_valid=true;out.landing_m=p;out.time_to_land_s=t;break;}prev=plane;}
    out.landing_sigma_m=std::sqrt(std::max({p6_(0,0),p6_(1,1),p6_(2,2)}));float sp=norm3(out.velocity_mps);auto fail=[&](const char*r){out.gate_reason=r;out.control_publish=false;};if(!gate_.enabled)fail("landing gate disabled");else if(!out.landing_valid)fail("no plane intersection");else if(track_frames_<trajectory_.min_track_frames)fail("track warming up");else if(out.confidence<gate_.min_confidence)fail("low confidence");else if(weight<gate_.min_student_weight)fail("Student-t outlier");else if(t<gate_.min_time_to_land_s||t>gate_.max_time_to_land_s)fail("impact time out of range");else if(sp<gate_.min_speed_mps)fail("speed too low");else if(out.landing_sigma_m>gate_.max_landing_sigma_m)fail("landing uncertainty");else if(std::abs(p.x)>gate_.max_abs_x_m||p.z<gate_.min_depth_m||p.z>gate_.max_depth_m)fail("landing outside range");else if(src!=ObservationSource::Rgbd&&!gate_.allow_fallback_observation)fail("fallback observation");else {float jump=norm3(p-last_landing_);stable_frames_=(stable_frames_==0||jump<=gate_.max_stable_jump_m)?stable_frames_+1:1;last_landing_=p;if(stable_frames_<gate_.stable_frames)fail("stabilizing");else{out.control_publish=true;out.gate_reason="publish";}}return out;
}

void drawTrajectoryStatus(cv::Mat& im,const BallTrackResult&r){if(!r.track_valid)return;char s[256];std::snprintf(s,sizeof(s),"TRACK %s v=%.2fm/s | LAND %s t=%.2fs | %s",observationSourceName(r.source),norm3(r.velocity_mps),r.landing_valid?"OK":"--",r.time_to_land_s,r.gate_reason.c_str());cv::rectangle(im,{8,im.rows-38},{std::min(im.cols-8,760),im.rows-8},{20,20,20},cv::FILLED);cv::putText(im,s,{16,im.rows-17},cv::FONT_HERSHEY_SIMPLEX,.55,r.control_publish?cv::Scalar(0,255,0):cv::Scalar(0,200,255),1,cv::LINE_AA);}
