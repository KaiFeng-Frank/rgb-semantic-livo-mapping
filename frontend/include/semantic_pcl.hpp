#pragma once
#include "semantic_frontend.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cstring>

namespace semantic_frontend {
// The pinned Velodyne non-feature path leaves the PCL normal channels unused.
// Preserve its point ABI and exact xyz/time arithmetic: normal_x carries class+1,
// normal_y confidence. NEVER average class IDs; reaggregate after VoxelGrid.
// Other sensor/feature modes explicitly reject semantic.enabled at startup.
template<class Point> Tag tagOf(const Point &p) {
  if(!std::isfinite(p.normal_x) || p.normal_x<1 || p.normal_x>kClasses ||
     p.normal_x!=std::floor(p.normal_x)) return {};
  Tag t{int(p.normal_x)-1,double(p.normal_y)};
  return t.valid()?t:Tag{};
}
template<class Point> void setTag(Point &p, Tag t) {
  p.normal_x=t.valid()?float(t.label+1):0.f;
  p.normal_y=t.valid()?float(t.confidence):0.f;
  p.normal_z=0.f;
}
class CloudTags {
  const sensor_msgs::msg::PointCloud2 &msg_;
  int label_=-1, confidence_=-1;
 public:
  explicit CloudTags(const sensor_msgs::msg::PointCloud2 &m, bool enabled=true):msg_(m) {
    if(!enabled) return;
    for(const auto &f:m.fields) {
      if(f.name=="semantic_label") {
        if(f.datatype!=sensor_msgs::msg::PointField::UINT16 || f.count!=1 || f.offset+2>m.point_step)
          throw std::invalid_argument("semantic_label must be one uint16");
        label_=int(f.offset);
      }
      if(f.name=="semantic_confidence") {
        if(f.datatype!=sensor_msgs::msg::PointField::FLOAT32 || f.count!=1 || f.offset+4>m.point_step)
          throw std::invalid_argument("semantic_confidence must be one float32");
        confidence_=int(f.offset);
      }
    }
    if((label_>=0)!=(confidence_>=0)) throw std::invalid_argument("incomplete semantic fields");
    if(label_>=0 && m.is_bigendian) throw std::invalid_argument("semantic input requires little endian");
  }
  Tag at(std::size_t i) const {
    if(label_<0) return {};
    if(!msg_.width || i>=std::size_t(msg_.width)*msg_.height) throw std::out_of_range("semantic point index");
    std::size_t base=(i/msg_.width)*msg_.row_step+(i%msg_.width)*msg_.point_step;
    if(base+msg_.point_step>msg_.data.size()) throw std::out_of_range("truncated semantic cloud");
    std::uint16_t label;float confidence;
    std::memcpy(&label,msg_.data.data()+base+label_,2);
    std::memcpy(&confidence,msg_.data.data()+base+confidence_,4);
    Tag t{int(label),double(confidence)};return t.valid()?t:Tag{};
  }
};
template<class Cloud,class Filter> void aggregateTags(const Cloud &raw, Cloud &down, const Filter &filter) {
  struct Votes { std::array<double,kClasses> mass{}; int count=0; };
  std::map<Cell,Votes> cells;
  auto key=[&filter](const auto &p)->Cell { auto c=filter.getGridCoordinates(p.x,p.y,p.z);return {c[0],c[1],c[2]}; };
  for(const auto &p:raw) {
    if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))continue;
    auto &v=cells[key(p)];++v.count;Tag t=tagOf(p);
    if(t.valid())v.mass[t.label]+=t.confidence;
  }
  for(auto &p:down) {
    auto it=cells.find(key(p));
    if(it==cells.end() || !it->second.count) {setTag(p,{});continue;}
    const auto &v=it->second;
    int c=int(std::max_element(v.mass.begin(),v.mass.end())-v.mass.begin());
    setTag(p,{c,v.mass[c]/v.count});
  }
}
} // namespace semantic_frontend
