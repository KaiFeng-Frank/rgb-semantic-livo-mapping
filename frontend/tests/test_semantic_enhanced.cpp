#include "semantic_likelihood.hpp"
#include "semantic_visual.hpp"
#include <Eigen/Geometry>
#include <iostream>
#include <stdexcept>
using namespace semantic_frontend;
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
std::vector<Sample> object(double x,int label=3) {
  std::vector<Sample> out;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    out.push_back({Vec3(x+i*.2,j*.2,5.),{label,.95}});
  return out;
}
void motionTests() {
  MotionTracker tracker;
  for(int frame=0;frame<8;++frame) {
    auto moving=object(frame*.6),parked=object(10.);
    auto world=moving;world.insert(world.end(),parked.begin(),parked.end());
    tracker.prepare(10.+frame*.1,world,0.);
    auto m=tracker.decision(moving[0]),p=tracker.decision(parked[0]);
    if(frame==0)require(m.state==Motion::Unknown && !m.map_allowed && m.weight>0.,"class alone was declared moving/static");
    if(frame>=4) {
      require(m.state==Motion::Moving && m.weight==0. && !m.map_allowed,"ego-compensated moving object remains in pose/map");
      require(p.state==Motion::Static && p.weight==1. && p.map_allowed,"parked car was removed by class");
      require(!tracker.invalidated.empty(),"old moving object map locations were not invalidated");
    }
    tracker.commit(Mat3::Identity(),Vec3::Zero());
  }
  require(!motionCandidate({7,.99}) && !motionCandidate({0,.99}) && motionCandidate({6,.99}),"native barrier/cone/pedestrian taxonomy mismatch");
  require(tracker.decision({Vec3::Zero(),{}}).weight==1.,"unknown labels changed baseline");
  bool threw=false;try{tracker.prepare(10.,{},0.);}catch(const std::invalid_argument&){threw=true;}
  require(threw,"backward motion time silently accepted");
  tracker.prepare(20.,object(10.),0.);
  require(tracker.decision(object(10.)[0]).state==Motion::Unknown,"stale parked track reused after long gap");
  // Large pose uncertainty must not masquerade as confident object motion.
  tracker.clear();
  for(int f=0;f<5;++f){auto o=object(f*.6);tracker.prepare(f*.1,o,2.);require(tracker.decision(o[0]).state!=Motion::Moving,"pose uncertainty ignored");tracker.commit(Mat3::Identity(),Vec3::Zero());}
}
void likelihoodTests() {
  std::vector<Sample> map;
  for(int x=-12;x<=12;++x)for(int y=-8;y<=8;++y)
    map.push_back({Vec3(x*.2,y*.2+.1,0.),{y<0?10:12,.95}});
  SemanticLikelihood field;field.commit(0,map,Mat3::Identity(),Vec3::Zero());
  require(field.linearize(1,map,Mat3::Identity(),Vec3(0,.1,0)).empty(),"single sweep treated as temporal support");
  field.commit(1,map,Mat3::Identity(),Vec3::Zero());
  auto rows=field.linearize(2,map,Mat3::Identity(),Vec3(0,.12,0));
  require(rows.size()>5,"categorical likelihood provides no pose observations");
  double rhs=0.,information=0.;
  for(const auto &r:rows){rhs-=r.information*r.jacobian[4]*r.residual;information+=r.information*r.jacobian.tail<3>().squaredNorm();}
  require(rhs<0.,"categorical alignment moves away from known semantic field");
  require(information<=field.information_cap+1e-8,"categorical information cap ineffective");
  const auto &r=rows[rows.size()/2];
  for(int j=0;j<6;++j) {
    const double h=1e-6;Vec3 plus=r.point_body,minus=plus;
    if(j<3){plus=Eigen::AngleAxisd(h,Vec3::Unit(j))*plus;minus=Eigen::AngleAxisd(-h,Vec3::Unit(j))*minus;}
    else{plus[j-3]+=h;minus[j-3]-=h;}
    auto residual=[&](Vec3 p){return std::sqrt(-2.*std::log(field.evaluate(2,p+Vec3(0,.12,0),r.first).probability));};
    require(std::abs((residual(plus)-residual(minus))/(2*h)-r.jacobian[j])<1e-5,"categorical pose Jacobian differs from finite differences");
  }
  auto shuffled=map;std::reverse(shuffled.begin(),shuffled.end());
  const auto shuffled_rows=field.linearize(2,shuffled,Mat3::Identity(),Vec3(0,.12,0));
  require(shuffled_rows.size()==rows.size(),"point ordering changed semantic coverage");
  for(std::size_t i=0;i<rows.size();++i)
    require(std::abs(shuffled_rows[i].residual-rows[i].residual)<1e-9 &&
            (shuffled_rows[i].jacobian-rows[i].jacobian).norm()<1e-8,"point ordering biases semantic observations");
  SemanticLikelihood uniform;for(auto &s:map)s.tag.label=10;
  uniform.commit(0,map,Mat3::Identity(),Vec3::Zero());uniform.commit(1,map,Mat3::Identity(),Vec3::Zero());
  require(uniform.linearize(2,map,Mat3::Identity(),Vec3(.1,0,0)).empty(),"uniform semantic field invents geometric constraints");
  bool threw=false;try{field.evaluate(1,Vec3::Zero(),10);}catch(const std::invalid_argument&){threw=true;}
  require(threw,"current semantic scan matched itself");
}
void visualTests() {
  VisualSemanticGate gate;ObjectRegion r;r.lo={-.5,-.5,4.5};r.hi={.5,.5,5.5};r.motion=Motion::Moving;
  auto project=[](const Vec3 &p)->Eigen::Vector2d{return {100.*p.x()/p.z()+80.,100.*p.y()/p.z()+60.};};
  gate.build(160,120,{}, {r},Mat3::Identity(),Vec3::Zero(),project);
  require(gate.moving_pixels>100 && gate.patchWeight({80,60},2)==0.,"moving pixels not masked");
  require(gate.patchWeight({30,30},2)==1.,"unrelated background masked");
  require(gate.patchWeight({64,60},2,2)==0.,"scaled patch crossing object boundary accepted");
  require(gate.patchWeight({1,1},2)==0.,"out-of-image patch permitted");
  r.motion=Motion::Static;gate.build(160,120,{}, {r},Mat3::Identity(),Vec3::Zero(),project);
  require(gate.moving_pixels==0 && gate.patchWeight({80,60},2)==1.,"parked car pixels removed");
  r.motion=Motion::Moving;
  gate.build(160,120,{{Vec3(0,0,2),{14,.99}}},{r},Mat3::Identity(),Vec3::Zero(),project);
  require(gate.weights.at<float>(60,80)==1.,"nearer static occluder overwritten by rear object mask");
  gate.clear();require(gate.patchWeight({80,60},2)==1.,"stale mask retained after clear");
}
int main(){motionTests();likelihoodTests();visualTests();std::cout<<"PASS temporal motion, categorical Jacobian, causal map, full visual patch masking\n";}
