#include "semantic_frontend.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <iostream>
#include <stdexcept>
using namespace semantic_frontend;
void check(bool b, const char *s) { if(!b) throw std::runtime_error(s); }
std::vector<Boundary> straight() {
  std::vector<Boundary> out;
  for(int i=-25;i<=25;++i) { Boundary b; b.point={i*.1,0.,0.}; b.confidence=.95; out.push_back(b); }
  return out;
}
int main() {
  Config cfg; cfg.enabled=true;
  check(group(8)==group(3) && group(7)!=group(3),"native trailer/cone class order changed");
  Evidence e;
  for(int i=0;i<100;++i) e.add({10,.95},0);
  check(!e.dominant().valid(),"pending evidence leaked into current estimate");
  e.flush(); check(e.frames==1,"one sweep counted as independent point observations");
  check(e.compatibility({14,.95})<e.compatibility({10,.95}),"semantic association has no effect");
  check(e.compatibility({-1,1.})==1.,"unknown query must be neutral");
  check(!Tag{3,std::numeric_limits<double>::quiet_NaN()}.valid(),"NaN confidence accepted");
  BoundaryMap map(cfg); const auto b=straight();
  map.commit(0,b,Mat3::Identity(),Vec3::Zero());
  check(map.linearize(1,b,Mat3::Identity(),Vec3(0,.2,0)).empty(),"single historical sweep accepted as stable");
  map.commit(1,b,Mat3::Identity(),Vec3::Zero());
  auto c=map.linearize(2,b,Mat3::Identity(),Vec3(0,.2,0));
  check(c.size()>10,"stable semantic boundary did not provide residuals");
  double info=0.; for(const auto &r:c) info+=r.information;
  check(info<=cfg.max_group_information+1e-8,"correlated group information not capped");
  // Finite differences use the same right SO(3) perturbation as FAST-LIVO2.
  const auto &q=c[c.size()/3];
  for(int j=0;j<6;++j) {
    const double h=1e-6;
    Vec3 rp=q.point_body,rm=q.point_body;
    if(j<3) {
      rp=Eigen::AngleAxisd(h,Vec3::Unit(j))*q.point_body;
      rm=Eigen::AngleAxisd(-h,Vec3::Unit(j))*q.point_body;
    } else { rp[j-3]+=h;rm[j-3]-=h; }
    check(std::abs(q.normal_world.dot(rp-rm)/(2*h)-q.jacobian[j])<1e-7,"incorrect semantic pose Jacobian");
  }
  // A flat ground plane constrains z but cannot recover a lateral y error.
  // The semantic road/sidewalk boundary must supply y information to the same solve.
  Eigen::Matrix<double,6,6> H=Eigen::Matrix<double,6,6>::Identity(); H(5,5)+=100.;
  Eigen::Matrix<double,6,1> rhs=Eigen::Matrix<double,6,1>::Zero();
  for(const auto &r:c) { H.noalias()+=r.information*r.jacobian.transpose()*r.jacobian; rhs.noalias()-=r.information*r.jacobian.transpose()*r.residual; }
  auto delta=H.ldlt().solve(rhs).eval();
  check(delta[4]<-.15 && std::abs(delta[5])<1e-6,"boundary failed to constrain geometric tangent direction");
  cfg.enabled=false; BoundaryMap disabled(cfg);
  disabled.commit(0,b,Mat3::Identity(),Vec3::Zero());disabled.commit(1,b,Mat3::Identity(),Vec3::Zero());
  check(disabled.linearize(2,b,Mat3::Identity(),Vec3::Zero()).empty(),"disabled mode adds information");
  auto wrong=b;for(auto &v:wrong) v.second=13;
  check(map.linearize(2,wrong,Mat3::Identity(),Vec3::Zero()).empty(),"different class pairs associated");
  bool threw=false;try { map.linearize(1,b,Mat3::Identity(),Vec3::Zero()); }catch(const std::invalid_argument&){threw=true;}
  check(threw,"current scan was allowed to match itself");
  std::vector<Sample> plane;
  for(int x=-20;x<=20;++x)for(int y=-8;y<=8;++y) plane.push_back({Vec3(x*.1,y*.1+.05,0.),{y<0?10:12,.95}});
  cfg.enabled=true; auto edges=extractBoundaries(plane,cfg);
  check(edges.size()>5,"semantic boundary extraction empty");
  for(const auto &v:edges) check(std::abs(v.point.y())<.06,"boundary midpoint biased away from transition");
  for(auto &p:plane) p.tag.label=10;
  check(extractBoundaries(plane,cfg).empty(),"uniform semantic field fabricated constraints");
  for(auto &p:plane) p.tag={6,.95};
  check(extractBoundaries(plane,cfg).empty(),"person used as permanent boundary landmark");
  for(int i=2;i<30;++i) map.commit(i,b,Mat3::Identity(),Vec3::Zero());
  check(map.size()<=cfg.history_frames*cfg.max_boundaries,"semantic history memory unbounded");
  std::cout << "PASS: association, causality, Jacobian, tangent observability, uncertainty cap, extraction, bounded map\n";
  std::cout << "synthetic lateral correction_m=" << delta[4] << " boundary_constraints=" << c.size() << " information=" << info << '\n';
}
