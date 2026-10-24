// Exercises the actual pinned FAST-LIVO2 StateEstimation, preprocessing, and PCL
// downsampling; this is an integration fixture, NOT a dataset accuracy result.
#include "voxel_map.h"
#include "preprocess.h"
#include "semantic_pcl.hpp"
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <stdexcept>
using namespace semantic_frontend;
void require(bool b,const char *s) {if(!b)throw std::runtime_error(s);}
struct Result {StatesGroup state;std::size_t rows;};
void checkAssociation() {
  VoxelMapConfig cfg{};cfg.max_voxel_size_=1.;cfg.max_layer_=0;cfg.sigma_num_=3.;
  std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> nodes;
  VoxelMapManager vm(cfg,nodes);
  for(int k=0;k<2;++k) {
    auto *n=new VoxelOctoTree(0,0,10,100,.01);
    auto &p=*n->plane_ptr_;p.is_plane_=true;p.center_={.2,.2,.2+.01*k};
    p.normal_=Vec3::UnitZ();p.d_=-p.center_.z();p.radius_=10.;p.plane_var_.setIdentity();p.plane_var_*=.001;
    n->semantic_evidence_.add({k?10:14,.95},0);n->semantic_evidence_.flush();
    vm.voxel_map_[VOXEL_LOCATION(0,k,0)]=n;
  }
  pointWithVar p;p.point_w=p.point_b=Vec3(.25,.2,.2);p.var=Mat3::Identity()*.05;p.body_var=Mat3::Identity()*.001;p.semantic_tag={10,.95};
  std::vector<pointWithVar> points{p};std::vector<PointToPlane> rows;
  vm.BuildResidualListOMP(points,rows);
  require(rows.size()==1 && std::abs(rows[0].center_.z()-.2)<1e-6,"geometry fixture chose unexpected surface");
  vm.semantic_map_.config.enabled=true;
  vm.BuildResidualListOMP(points,rows);
  require(rows.size()==1 && std::abs(rows[0].center_.z()-.21)<1e-6,"semantic conflict did not change actual surface association");
  for(auto &[key,node]:vm.voxel_map_)delete node;
}
void checkPreprocessing() {
  auto msg=std::make_shared<sensor_msgs::msg::PointCloud2>();
  msg->height=1;msg->width=6;msg->point_step=32;msg->row_step=192;msg->is_dense=true;msg->data.resize(192);
  auto field=[&](std::string name,int offset,int type) {sensor_msgs::msg::PointField f;f.name=name;f.offset=offset;f.datatype=type;f.count=1;msg->fields.push_back(f);};
  field("x",0,7);field("y",4,7);field("z",8,7);field("intensity",12,7);field("time",16,7);field("ring",20,4);
  field("semantic_label",24,4);field("semantic_confidence",28,7);
  for(int i=0;i<6;++i) {
    float v[5]={float(i+3),.1f,-1.f,.5f,float(i*10000)};
    std::memcpy(msg->data.data()+i*32,v,20);
    std::uint16_t ring=i,label=i%2?12:10;float confidence=.95f;
    std::memcpy(msg->data.data()+i*32+20,&ring,2);std::memcpy(msg->data.data()+i*32+24,&label,2);std::memcpy(msg->data.data()+i*32+28,&confidence,4);
  }
  Preprocess off,on;off.set(false,VELO16,.01,1);on.set(false,VELO16,.01,1);off.N_SCANS=on.N_SCANS=64;on.semantic_enabled=true;
  PointCloudXYZI::Ptr a(new PointCloudXYZI),b(new PointCloudXYZI);
  off.process(msg,a);on.process(msg,b);require(a->size()==6 && b->size()==6,"preprocessing changed point coverage");
  for(std::size_t i=0;i<a->size();++i) {
    require(((*a)[i].getVector3fMap()-(*b)[i].getVector3fMap()).norm()==0. && (*a)[i].curvature==(*b)[i].curvature,"semantic preprocessing changed xyz/time");
    require(tagOf((*b)[i]).label==(i%2?12:10),"raw semantic point identity lost");
  }
}
Result estimate(bool enabled,bool boundaries,bool coarse_geometry=false) {
  VoxelMapConfig cfg{};
  cfg.max_voxel_size_=20.;cfg.max_layer_=0;cfg.max_iterations_=6;
  cfg.layer_init_num_={10};cfg.max_points_num_=5000;cfg.planner_threshold_=.01;
  cfg.beam_err_=.05;cfg.dept_err_=.03;cfg.sigma_num_=3.;
  std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> voxels;
  VoxelMapManager vm(cfg,voxels);
  vm.extR_=M3D::Identity();vm.extT_=V3D::Zero();
  vm.state_.cov=Eigen::Matrix<double,DIM_STATE,DIM_STATE>::Identity()*.1;
  vm.state_.cov(3,3)=.01;vm.state_.cov(5,5)=.01;
  // Pose-velocity cross covariance must carry the semantic innovation to velocity.
  vm.state_.cov(4,8)=vm.state_.cov(8,4)=.02;
  std::vector<Sample> points;
  for(int x=-30;x<=30;++x)for(int y=-10;y<=10;++y) {
    PointType p;p.x=x*.1f;p.y=y*.1f+.05f;p.z=-1.;p.intensity=1.;p.curvature=0.;
    setTag(p,{y<0?10:12,.95});vm.feats_down_body_->push_back(p);
    vm.feats_down_world_->push_back(p);points.push_back({Vec3(p.x,p.y,p.z),tagOf(p)});
  }
  vm.feats_undistort_=vm.feats_down_body_;
  if (coarse_geometry) {
    PointCloudXYZI::Ptr geometry(new PointCloudXYZI);
    pcl::VoxelGrid<PointType> filter;filter.setLeafSize(.5,.5,.5);
    filter.setInputCloud(vm.feats_undistort_);filter.filter(*geometry);
    aggregateTags(*vm.feats_undistort_,*geometry,filter);
    vm.feats_down_body_=geometry;*vm.feats_down_world_=*geometry;
  }
  vm.feats_down_size_=int(vm.feats_down_body_->size());
  vm.BuildVoxelMap();
  vm.semantic_map_.config.enabled=enabled;vm.semantic_map_.config.boundaries=boundaries;
  auto bs=extractBoundaries(points,vm.semantic_map_.config);
  vm.semantic_map_.commit(0,bs,M3D::Identity(),V3D::Zero());
  vm.semantic_map_.commit(1,bs,M3D::Identity(),V3D::Zero());
  vm.semantic_frame_=2;vm.state_.pos_end.y()=.2;
  StatesGroup prior=vm.state_;
  vm.StateEstimation(prior);
  auto result=Result{vm.state_,vm.semantic_constraint_count_};
  for(auto &[key,node]:vm.voxel_map_)delete node;
  return result;
}
int main(int argc,char **argv) {
  rclcpp::init(argc,argv);
  checkPreprocessing();checkAssociation();
  const auto off=estimate(false,false),neutral=estimate(true,false),on=estimate(true,true);
  std::cout<<"integration diagnostic off_y="<<off.state.pos_end.y()<<" on_y="<<on.state.pos_end.y()<<" rows="<<on.rows<<std::endl;
  require((off.state.pos_end-neutral.state.pos_end).norm()<1e-12,"neutral semantic path changed baseline pose");
  require((off.state.cov-neutral.state.cov).norm()<1e-12,"neutral semantic path changed baseline covariance");
  require(std::abs(off.state.pos_end.y()-.2)<1e-6,"fixture not geometrically degenerate in y");
  require(on.rows>5 && std::abs(on.state.pos_end.y())<.10,"actual FAST-LIVO2 did not consume semantic residuals");
  require(on.state.cov(4,4)<off.state.cov(4,4),"semantic rows omitted from covariance update");
  require(std::abs(on.state.vel_end.y())>.001,"semantic innovation did not reach coupled velocity state");
  const auto coarse_off=estimate(false,false,true),coarse_on=estimate(true,true,true);
  require(std::abs(coarse_off.state.pos_end.y()-.2)<1e-6,"coarse geometry fixture has lateral information");
  require(coarse_on.rows>5 && std::abs(coarse_on.state.pos_end.y())<.10,
          "geometry voxelization erased the separate dense semantic boundary input");
  // PCL averages normals; the adapter must restore discrete tags without changing xyz.
  PointCloudXYZI::Ptr cloud(new PointCloudXYZI);
  for(int i=0;i<5;++i) {PointType p;p.x=.1+.01*i;p.y=.1;p.z=.1;p.intensity=i;p.curvature=i;setTag(p,{i<4?10:12,.95});cloud->push_back(p);}
  pcl::VoxelGrid<PointType> filter;filter.setLeafSize(1.,1.,1.);filter.setInputCloud(cloud);
  PointCloudXYZI down;filter.filter(down);auto before=down;
  aggregateTags(*cloud,down,filter);
  require(down.size()==1 && tagOf(down[0]).label==10,"voxel tags were numerically averaged");
  require((down[0].getVector3fMap()-before[0].getVector3fMap()).norm()==0.,"semantic aggregation moved geometry");
  require(std::abs(tagOf(down[0]).confidence-.76)<1e-6,"mixed semantic voxel overconfident");
  std::cout<<"PASS actual FAST-LIVO2: off_y="<<off.state.pos_end.y()<<" on_y="<<on.state.pos_end.y()
           <<" coupled_vy="<<on.state.vel_end.y()<<" boundary_rows="<<on.rows
           <<" off_variance="<<off.state.cov(4,4)<<" on_variance="<<on.state.cov(4,4)<<std::endl;
  rclcpp::shutdown();
}
