#pragma once
#include "semantic_motion.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <functional>

namespace semantic_frontend {
// Image-space gate projected from causal 3-D motion regions. This is not a dense
// 2-D segmentation model. Nearer observed static surfaces protect their pixels.
class VisualSemanticGate {
 public:
  cv::Mat weights;
  std::size_t moving_pixels=0, uncertain_pixels=0;
  void clear(){weights.release();moving_pixels=uncertain_pixels=0;}
  void build(int width,int height,const std::vector<Sample> &world,
             const std::vector<ObjectRegion> &regions,const Mat3 &Rcw,const Vec3 &Pcw,
             const std::function<Eigen::Vector2d(const Vec3&)> &project) {
    weights=cv::Mat(height,width,CV_32F,cv::Scalar(1.));
    cv::Mat depth(height,width,CV_32F,cv::Scalar(std::numeric_limits<float>::infinity()));
    for(const auto &s:world) {
      if(!s.point.allFinite() || motionCandidate(s.tag))continue;
      Vec3 p=Rcw*s.point+Pcw;if(p.z()<.1)continue;
      Eigen::Vector2d uv=project(p);if(!uv.allFinite())continue;
      if(uv.x()<0 || uv.y()<0 || uv.x()>=width || uv.y()>=height)continue;
      const int x=int(uv.x()),y=int(uv.y());
      for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx)
        if(x+dx>=0&&x+dx<width&&y+dy>=0&&y+dy<height)
          depth.at<float>(y+dy,x+dx)=std::min(depth.at<float>(y+dy,x+dx),float(p.z()));
    }
    for(const auto &r:regions) {
      if(r.motion==Motion::Static)continue;
      std::vector<cv::Point> polygon;double nearest=std::numeric_limits<double>::infinity();bool behind=false;
      for(int i=0;i<8;++i) {
        Vec3 corner;for(int j=0;j<3;++j)corner[j]=(i&(1<<j))?r.hi[j]+.1:r.lo[j]-.1;
        Vec3 p=Rcw*corner+Pcw;
        if(p.z()<.1){behind=true;break;}
        nearest=std::min(nearest,p.z());Eigen::Vector2d uv=project(p);
        if(!uv.allFinite()){behind=true;break;}
        polygon.emplace_back(int(std::clamp(uv.x(),-double(width),2.*width)),
                             int(std::clamp(uv.y(),-double(height),2.*height)));
      }
      if(behind)continue; // do not let a box through the camera fill the image
      std::vector<cv::Point> hull;cv::convexHull(polygon,hull);
      cv::Mat mask(height,width,CV_8U,cv::Scalar(0));cv::fillConvexPoly(mask,hull,cv::Scalar(255));
      const float value=r.motion==Motion::Moving?0.f:.25f;
      for(int y=0;y<height;++y)for(int x=0;x<width;++x)
        if(mask.at<unsigned char>(y,x) && depth.at<float>(y,x)>=nearest-.2)
          weights.at<float>(y,x)=std::min(weights.at<float>(y,x),value);
    }
    moving_pixels=cv::countNonZero(weights==0.f);
    uncertain_pixels=cv::countNonZero((weights>0.f)&(weights<1.f));
  }
  // Gate the complete patch, including interpolation/gradient support at its
  // actual pyramid/search scale, rather than testing only its centre pixel.
  double patchWeight(const Eigen::Vector2d &uv,int half,int scale=1) const {
    if(weights.empty())return 1.;
    if(!uv.allFinite() || scale<1 || half<0)return 0.;
    const int margin=(half+2)*scale;
    if(uv.x()<margin || uv.y()<margin || uv.x()+margin>=weights.cols || uv.y()+margin>=weights.rows)return 0.;
    const int x=int(std::floor(uv.x()/scale))*scale,y=int(std::floor(uv.y()/scale))*scale;
    const int x0=x-margin,y0=y-margin,x1=x+margin,y1=y+margin;
    if(x0<0||y0<0||x1>=weights.cols||y1>=weights.rows)return 0.;
    double minimum=1.;cv::minMaxLoc(weights(cv::Rect(x0,y0,x1-x0+1,y1-y0+1)),&minimum,nullptr);
    return minimum;
  }
};
} // namespace semantic_frontend
