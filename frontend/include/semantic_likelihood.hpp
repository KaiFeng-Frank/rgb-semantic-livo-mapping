#pragma once
#include "semantic_frontend.hpp"
#include <set>

namespace semantic_frontend {
// A pose-dependent categorical observation, independent of boundary extraction.
// Probabilities are smoothed historical class-vote frequencies, NOT reconstructed
// network logits. Only observed static classes and older scans enter the field.
class SemanticLikelihood {
  struct Node {Vec3 point=Vec3::Zero();std::array<double,kClasses> votes{};double count=0.;};
  struct Frame {std::int64_t id;std::map<Cell,Node> nodes;};
  std::deque<Frame> frames_;
 public:
  double cell_size=.4, radius=1.2, information_cap=20.;
  std::size_t history=12, max_nodes=5000;
  struct Value {double probability=0.;Vec3 gradient=Vec3::Zero();int supports=0;};
  void clear(){frames_.clear();}
  std::size_t size() const {std::size_t n=0;for(const auto &f:frames_)n+=f.nodes.size();return n;}
  void commit(std::int64_t frame,const std::vector<Sample> &body,const Mat3 &R,const Vec3 &t) {
    if(!frames_.empty() && frame<=frames_.back().id)throw std::invalid_argument("likelihood frame order");
    Frame f{frame,{}};
    for(const auto &s:body) {
      if(!s.tag.valid() || s.tag.label<10 || s.tag.confidence<.8 || !s.point.allFinite() || s.point.norm()>70.)continue;
      const Vec3 p=R*s.point+t;const auto key=cell(p,cell_size);
      if(f.nodes.size()>=max_nodes && !f.nodes.count(key))continue;
      auto &n=f.nodes[key];n.point+=p;n.votes[s.tag.label]+=s.tag.confidence;n.count+=1.;
    }
    for(auto &[key,n]:f.nodes){n.point/=n.count;double total=0.;for(double v:n.votes)total+=v;for(double &v:n.votes)v/=total;}
    frames_.push_back(std::move(f));while(frames_.size()>history)frames_.pop_front();
  }
  Value evaluate(std::int64_t frame,const Vec3 &world,int label) const {
    if(!frames_.empty() && frame<=frames_.back().id)throw std::invalid_argument("self/future semantic likelihood");
    Value out;if(label<10 || label>=kClasses || !world.allFinite())return out;
    double sum=0.,correct=0.;Vec3 ds=Vec3::Zero(),dc=Vec3::Zero();
    auto [x,y,z]=cell(world,cell_size);const int reach=int(std::ceil(radius/cell_size));
    for(const auto &f:frames_) {
      double mass=0.;
      for(int dx=-reach;dx<=reach;++dx)for(int dy=-reach;dy<=reach;++dy)for(int dz=-reach;dz<=reach;++dz) {
        auto it=f.nodes.find({x+dx,y+dy,z+dz});if(it==f.nodes.end())continue;
        const auto &n=it->second;const Vec3 d=world-n.point;const double u=d.norm()/radius;
        if(u>=1.)continue;
        const double w=std::pow(1.-u,4)*(1.+4.*u);
        const Vec3 dw=-20.*std::pow(1.-u,3)/(radius*radius)*d;
        sum+=w;correct+=w*n.votes[label];ds+=dw;dc+=dw*n.votes[label];mass+=w;
      }
      out.supports+=mass>.05;
    }
    if(out.supports<2 || sum<.1)return {};
    out.probability=.01+.98*correct/sum;
    out.gradient=.98*(dc*sum-correct*ds)/(sum*sum);
    return out;
  }
  std::vector<Constraint> linearize(std::int64_t frame,const std::vector<Sample> &body,const Mat3 &R,const Vec3 &t) const {
    std::vector<Constraint> out;
    struct Query {Vec3 sum=Vec3::Zero();double confidence=0.;int n=0,label=-1;};
    std::map<std::tuple<Cell,int>,Query> queries;
    for(const auto &s:body) {
      if(!s.tag.valid() || s.tag.label<10 || s.tag.confidence<.8 || !s.point.allFinite())continue;
      auto &q=queries[{cell(s.point,.8),s.tag.label}];
      q.sum+=s.point;q.confidence+=s.tag.confidence;++q.n;q.label=s.tag.label;
    }
    // Cell/class centroids avoid the order-dependent first-point bias at edges.
    const std::size_t stride=std::max<std::size_t>(1,(queries.size()+255)/256);
    std::size_t index=0;
    for(const auto &[key,q]:queries) {
      if(index++%stride)continue;
      const Sample s{q.sum/double(q.n),{q.label,q.confidence/q.n}};
      auto v=evaluate(frame,R*s.point+t,s.tag.label);
      if(v.supports<2 || v.gradient.norm()<1e-4 || v.probability<.05)continue;
      Constraint c;c.point_body=s.point;c.first=s.tag.label;c.second=s.tag.label;
      c.residual=std::sqrt(-2.*std::log(v.probability));
      Vec3 derivative=-v.gradient/(v.probability*c.residual);
      c.jacobian.head<3>()=-derivative.transpose()*R*skew(s.point);
      c.jacobian.tail<3>()=derivative.transpose();
      c.information=reliability(s.tag)*std::min(1.,1./c.residual);
      out.push_back(c);if(out.size()>=256)break;
    }
    // Cap the summed translational information, not merely the number of points.
    double total=0.;for(const auto &c:out)total+=c.information*c.jacobian.tail<3>().squaredNorm();
    for(auto &c:out)c.information*=std::min(1.,information_cap/std::max(total,1e-12));
    return out;
  }
};
} // namespace semantic_frontend
