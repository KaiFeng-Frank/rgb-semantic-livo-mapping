#pragma once
#include "semantic_frontend.hpp"
#include <numeric>
#include <set>

namespace semantic_frontend {
// A class is only a candidate. Motion is decided from ego-compensated,
// temporally separated observations, before the current estimator update.
enum class Motion { Unknown, Static, Moving };
inline bool motionCandidate(Tag t) {
  // Native nuScenes: bicycle, bus, car, construction vehicle, motorcycle,
  // pedestrian, trailer and truck (same order as src/sem_core.py).
  // Barriers and traffic cones are not assumed to move.
  return t.valid() && t.confidence >= .8 &&
      (t.label==1 || t.label==2 || t.label==3 || t.label==4 ||
       t.label==5 || t.label==6 || t.label==8 || t.label==9);
}
struct ObjectRegion {
  Vec3 center=Vec3::Zero(), lo=Vec3::Zero(), hi=Vec3::Zero();
  Vec3 anchor=Vec3::Zero();
  int label=-1, id=-1, still=0, moving=0;
  double stamp=0., anchor_stamp=0., variance=0.;
  Motion motion=Motion::Unknown;
  bool contains(const Vec3 &p, double padding=.25) const {
    return (p.array() >= lo.array()-padding).all() && (p.array() <= hi.array()+padding).all();
  }
};
struct MotionDecision {
  double weight=1.;
  bool map_allowed=true;
  Motion state=Motion::Static;
};
class MotionTracker {
  std::vector<ObjectRegion> tracks_, pending_;
  int next_id_=0;
  double last_stamp_=-1.;
 public:
  std::vector<ObjectRegion> invalidated;
  const std::vector<ObjectRegion> &regions() const {return pending_;}
  void clear() {tracks_.clear();pending_.clear();invalidated.clear();last_stamp_=-1.;next_id_=0;}
  void prepare(double stamp, const std::vector<Sample> &world, double pose_variance) {
    if (!std::isfinite(stamp) || stamp<=last_stamp_) throw std::invalid_argument("motion timestamps must increase");
    if (!std::isfinite(pose_variance) || pose_variance<0.) throw std::invalid_argument("invalid pose variance");
    if(last_stamp_>=0. && stamp-last_stamp_>1.) tracks_.clear();
    pending_.clear();invalidated.clear();
    std::vector<int> ids;
    for(std::size_t i=0;i<world.size();++i)
      if(world[i].point.allFinite() && motionCandidate(world[i].tag)) ids.push_back(int(i));
    std::vector<int> parent(ids.size());std::iota(parent.begin(),parent.end(),0);
    auto root=[&](int n) {while(parent[n]!=n){parent[n]=parent[parent[n]];n=parent[n];}return n;};
    std::map<Cell,std::vector<int>> bins;
    for(int k=0;k<int(ids.size());++k) {
      const auto &q=world[ids[k]]; auto [x,y,z]=cell(q.point,.8);
      for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy)for(int dz=-1;dz<=1;++dz) {
        auto it=bins.find({x+dx,y+dy,z+dz});if(it==bins.end())continue;
        for(int j:it->second)
          if(q.tag.label==world[ids[j]].tag.label && (q.point-world[ids[j]].point).squaredNorm()<.64)
            parent[root(k)]=root(j);
      }
      bins[{x,y,z}].push_back(k);
    }
    std::map<int,std::vector<int>> groups;
    for(int k=0;k<int(ids.size());++k)groups[root(k)].push_back(ids[k]);
    std::set<int> used;
    for(const auto &[key,indices]:groups) {
      if(pending_.size()>=128)continue;
      ObjectRegion q;q.lo=q.hi=world[indices[0]].point;q.label=world[indices[0]].tag.label;
      for(int i:indices){q.center+=world[i].point;q.lo=q.lo.cwiseMin(world[i].point);q.hi=q.hi.cwiseMax(world[i].point);}
      q.center/=double(indices.size());q.stamp=stamp;
      // View-dependent centroid changes have a non-vanishing uncertainty floor.
      q.variance=pose_variance+.04;
      double best=std::numeric_limits<double>::infinity(),second=best;int match=-1;
      for(int i=0;i<int(tracks_.size());++i) {
        const auto &t=tracks_[i];
        if(t.label!=q.label || used.count(t.id) || stamp-t.stamp>1.)continue;
        const double shape=(q.hi-q.lo-(t.hi-t.lo)).norm();
        if(shape>1.)continue; // split, occlusion or merge: do not infer motion
        const double d=(q.center-t.center).norm();
        if(d<best){second=best;best=d;match=i;}else second=std::min(second,d);
      }
      if(indices.size()>=4 && match>=0 && best<1.5+8.*(stamp-tracks_[match].stamp) && second-best>.3) {
        const auto &t=tracks_[match];used.insert(t.id);
        q.id=t.id;q.anchor=t.anchor;q.anchor_stamp=t.anchor_stamp;q.still=t.still;q.moving=t.moving;
        const double elapsed=stamp-q.anchor_stamp;
        const double displacement=(q.center-q.anchor).norm();
        const double sigma=std::sqrt(q.variance+t.variance);
        if(elapsed>=.2) {
          if(displacement>std::max(.25,3.*sigma) && displacement/elapsed>.5) {++q.moving;q.still=0;}
          else if(displacement<.15 && best<.15) {++q.still;q.moving=0;}
          else {q.still=0;q.moving=0;}
        }
        if(q.moving>=2 || (t.motion==Motion::Moving && displacement>.2))q.motion=Motion::Moving;
        else if(q.still>=2)q.motion=Motion::Static;
        if(q.motion==Motion::Moving){
          invalidated.push_back(t);invalidated.push_back(q);
          ObjectRegion anchor=t;const Vec3 offset=t.anchor-t.center;
          anchor.center+=offset;anchor.lo+=offset;anchor.hi+=offset;
          invalidated.push_back(anchor);
        }
        // Keep a finite temporal baseline; a parked object can start moving later.
        if(elapsed>1. && q.motion==Motion::Static){q.anchor=q.center;q.anchor_stamp=stamp;}
      } else {q.id=next_id_++;q.anchor=q.center;q.anchor_stamp=stamp;}
      pending_.push_back(q);
    }
    last_stamp_=stamp;
  }
  MotionDecision decision(const Sample &s) const {
    if(!motionCandidate(s.tag))return {};
    for(const auto &q:pending_)if(q.label==s.tag.label && q.contains(s.point)) {
      if(q.motion==Motion::Moving)return {0.,false,Motion::Moving};
      if(q.motion==Motion::Static)return {};
      return {.25,false,Motion::Unknown};
    }
    return {.25,false,Motion::Unknown}; // insufficient support is not static evidence
  }
  // Called once after estimation; current decisions remain fixed during IEKF.
  void commit(const Mat3 &correction, const Vec3 &translation) {
    for(auto &q:pending_) {
      const Vec3 old=q.center;q.center=correction*q.center+translation;
      Vec3 lo=Vec3::Constant(std::numeric_limits<double>::infinity()),hi=-lo;
      for(int i=0;i<8;++i) {
        Vec3 v;for(int j=0;j<3;++j)v[j]=(i&(1<<j))?q.hi[j]:q.lo[j];
        v=correction*v+translation;lo=lo.cwiseMin(v);hi=hi.cwiseMax(v);
      }
      q.lo=lo;q.hi=hi;
      if(q.anchor_stamp==q.stamp)q.anchor=q.center;
      // Do not move older anchors with the current pose correction.
      q.variance+=(q.center-old).squaredNorm()/3.;
    }
    tracks_=pending_;
  }
};
} // namespace semantic_frontend
