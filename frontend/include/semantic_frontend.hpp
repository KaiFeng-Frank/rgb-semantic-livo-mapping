// Semantic observations for the FAST-LIVO2 error-state estimator.
// No ROS, trained model, ground truth, or pose optimizer lives in this module.
#pragma once
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace semantic_frontend {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Row6 = Eigen::Matrix<double, 1, 6>;
constexpr int kClasses = 16;
constexpr int kUnknown = -1;

struct Tag {
  int label = kUnknown;
  double confidence = 0.;
  bool valid() const {
    return label >= 0 && label < kClasses && std::isfinite(confidence) &&
           confidence >= 0. && confidence <= 1.;
  }
};
inline bool movable(int c) { return c >= 1 && c <= 9; }
inline bool ground(int c) { return c >= 10 && c <= 13; }
inline double reliability(Tag t) {
  return t.valid() ? std::clamp((t.confidence - .5) / .5, 0., 1.) : 0.;
}
inline int group(int c) {
  if (c == 1 || c == 5) return 1;
  if (c == 2 || c == 3 || c == 4 || c == 8 || c == 9) return 2;
  if (c == 6) return 3;
  if (ground(c)) return 4;
  if (c == 15) return 5;
  return 0;
}
inline Mat3 skew(const Vec3 &p) {
  Mat3 s;
  s << 0., -p.z(), p.y(), p.z(), 0., -p.x(), -p.y(), p.x(), 0.;
  return s;
}

// Correlated samples from one sweep contribute at most one vote per surface.
// Commit ONLY after the state update. Unknown observations are neutral.
struct Evidence {
  std::array<double, kClasses> mass{};
  std::array<double, kClasses> pending{};
  std::int64_t last_frame = -1;
  int frames = 0;
  void add(Tag tag, std::int64_t frame) {
    if (!tag.valid() || reliability(tag) == 0.) return;
    if (frame < last_frame) throw std::invalid_argument("semantic frame moved backwards");
    if (frame != last_frame) {
      flush();
      last_frame = frame;
    }
    pending[tag.label] += reliability(tag);
  }
  void flush() {
    double total = 0.;
    for (double v : pending) total += v;
    if (total == 0.) return;
    for (int c = 0; c < kClasses; ++c) {
      mass[c] = .98 * mass[c] + pending[c] / total;
      pending[c] = 0.;
    }
    ++frames;
  }
  Tag dominant() const {
    double sum = 0.;
    for (double v : mass) sum += v;
    if (sum <= 0.) return {};
    int c = int(std::max_element(mass.begin(), mass.end()) - mass.begin());
    return {c, mass[c] / sum};
  }
  double compatibility(Tag query) const {
    const Tag target = dominant();
    const double r = reliability(query) * reliability(target);
    if (r == 0.) return 1.;
    const double match = query.label == target.label ? 1. :
                         group(query.label) == group(target.label) ? .7 : .2;
    return 1. - r * (1. - match);
  }
};

struct Config {
  bool enabled = false;
  bool association = true;
  bool boundaries = true;
  double confidence_min = .8;
  double extraction_radius = .65;
  double correspondence_radius = .8;
  double boundary_sigma = .20; // metres; includes segmentation/sampling uncertainty
  double huber = .30;
  double max_frame_information = 100.; // cap correlated boundary observations
  double max_group_information = 25.;
  int min_support = 3;
  int history_frames = 12;
  int min_history_frames = 2;
  std::size_t max_boundaries = 1000;
  double max_map_radius = 70.;
};

struct Sample { Vec3 point; Tag tag; };
struct Boundary {
  Vec3 point = Vec3::Zero();
  Vec3 surface_normal = Vec3::UnitZ();
  int first = 10, second = 12;
  double confidence = 0.;
  bool samePair(const Boundary &b) const { return first == b.first && second == b.second; }
};
using Cell = std::tuple<int, int, int>;
inline Cell cell(const Vec3 &p, double size) {
  return {int(std::floor(p.x()/size)), int(std::floor(p.y()/size)), int(std::floor(p.z()/size))};
}

// Initial scope: boundaries between observed ground categories. A locally planar
// neighbourhood rejects range silhouettes and height discontinuities. It does NOT
// create a pole/lane class that is absent from the native 16-class model.
inline std::vector<Boundary> extractBoundaries(const std::vector<Sample> &points, const Config &cfg) {
  std::map<Cell, std::vector<int>> grid;
  for (std::size_t i = 0; i < points.size(); ++i)
    if (points[i].point.allFinite() && ground(points[i].tag.label) &&
        points[i].tag.valid() && points[i].tag.confidence >= cfg.confidence_min)
      grid[cell(points[i].point, cfg.extraction_radius)].push_back(int(i));
  std::vector<Boundary> result;
  std::map<std::tuple<int,int,int,int,int>, bool> emitted;
  for (const auto &[key, ids] : grid) for (int i : ids) {
    const auto &p = points[i];
    std::vector<int> near;
    auto [x,y,z] = key;
    for (int dx=-1; dx<=1; ++dx) for (int dy=-1; dy<=1; ++dy) for (int dz=-1; dz<=1; ++dz) {
      auto it = grid.find({x+dx,y+dy,z+dz});
      if (it == grid.end()) continue;
      for (int j : it->second)
        if ((points[j].point-p.point).squaredNorm() <= cfg.extraction_radius*cfg.extraction_radius) near.push_back(j);
    }
    int other = -1, same_count = 0;
    double spacing = std::numeric_limits<double>::infinity();
    double best = std::numeric_limits<double>::infinity();
    for (int j : near) {
      if (points[j].tag.label == p.tag.label) {
        ++same_count;
        if (j!=i) spacing=std::min(spacing,(points[j].point-p.point).squaredNorm());
        continue;
      }
      double d = (points[j].point-p.point).squaredNorm();
      if (d < best) { best=d; other=j; }
    }
    // Only an adjacent class transition is an edge. Interior points within the
    // search radius would otherwise smear the boundary across a wide strip.
    if (other < 0 || same_count < cfg.min_support || best>2.25*spacing) continue;
    int other_count=0;
    Vec3 mean=Vec3::Zero();
    for (int j : near) {
      mean += points[j].point;
      other_count += points[j].tag.label == points[other].tag.label;
    }
    if (other_count < cfg.min_support) continue;
    mean /= double(near.size());
    Mat3 cov=Mat3::Zero();
    for (int j : near) { Vec3 d=points[j].point-mean; cov.noalias() += d*d.transpose(); }
    cov /= double(near.size());
    Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
    if (es.info()!=Eigen::Success || es.eigenvalues()[1] < 1e-4 ||
        es.eigenvalues()[0] > .04*es.eigenvalues()[1]) continue;
    Boundary b;
    b.point=.5*(p.point+points[other].point);
    b.surface_normal=es.eigenvectors().col(0);
    b.first=std::min(p.tag.label,points[other].tag.label);
    b.second=std::max(p.tag.label,points[other].tag.label);
    b.confidence=std::min(p.tag.confidence,points[other].tag.confidence);
    auto [bx,by,bz]=cell(b.point,.25);
    if (!emitted.emplace(std::make_tuple(bx,by,bz,b.first,b.second),true).second) continue;
    result.push_back(b);
    if (result.size() >= cfg.max_boundaries) return result;
  }
  return result;
}

struct Constraint {
  Vec3 point_body, normal_world, center_world;
  double residual=0., information=0.;
  Row6 jacobian=Row6::Zero();
  int first=0, second=0;
};

// All map observations are older than the current frame; the map is immutable
// throughout IEKF iterations. A line fit to same-pair historical boundary points
// supplies ONE tangential residual, leaving plane-normal geometry to the LIO.
class BoundaryMap {
  struct Frame { std::int64_t id; std::vector<Boundary> observations; };
  std::deque<Frame> frames_;
 public:
  Config config;
  explicit BoundaryMap(Config c={}) : config(c) {}
  void clear() { frames_.clear(); }
  std::size_t size() const {
    std::size_t n=0; for (const auto &f:frames_) n+=f.observations.size(); return n;
  }
  void commit(std::int64_t id, const std::vector<Boundary> &body, const Mat3 &R, const Vec3 &t) {
    if (!frames_.empty() && id<=frames_.back().id) throw std::invalid_argument("duplicate/out-of-order map commit");
    Frame f{id,{}};
    for (auto b:body) {
      if (!b.point.allFinite() || !b.surface_normal.allFinite() || b.point.norm()>config.max_map_radius) continue;
      b.point=R*b.point+t; b.surface_normal=R*b.surface_normal;
      f.observations.push_back(b);
      if(f.observations.size()>=config.max_boundaries) break;
    }
    frames_.push_back(std::move(f));
    while (int(frames_.size())>config.history_frames) frames_.pop_front();
  }
  std::vector<Constraint> linearize(std::int64_t frame, const std::vector<Boundary> &query,
                                    const Mat3 &R, const Vec3 &t) const {
    std::vector<Constraint> out;
    if (!config.enabled || !config.boundaries) return out;
    if (!frames_.empty() && frames_.back().id>=frame) throw std::invalid_argument("current/future scan in semantic map");
    for(const auto &b:query) {
      const Vec3 world=R*b.point+t;
      std::vector<Vec3> nearby;
      Vec3 historical_normal=Vec3::Zero();
      int supports=0;
      for(const auto &f:frames_) {
        bool seen=false;
        for(const auto &m:f.observations) {
          if (!b.samePair(m) || std::abs((R*b.surface_normal).dot(m.surface_normal))<.9 ||
              (m.point-world).norm()>config.correspondence_radius) continue;
          nearby.push_back(m.point); seen=true;
          historical_normal += (R*b.surface_normal).dot(m.surface_normal)<0. ? -m.surface_normal : m.surface_normal;
        }
        supports+=seen;
      }
      if(supports<config.min_history_frames || nearby.size()<6) continue;
      Vec3 center=Vec3::Zero(); for(const auto &p:nearby) center+=p;
      center/=double(nearby.size());
      Mat3 cov=Mat3::Zero(); for(const auto &p:nearby) { Vec3 d=p-center; cov.noalias()+=d*d.transpose(); }
      cov/=double(nearby.size());
      Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
      if(es.info()!=Eigen::Success || es.eigenvalues()[2]<.01 || es.eigenvalues()[1]>.15*es.eigenvalues()[2]) continue;
      if(historical_normal.norm()<1e-6) continue;
      Vec3 normal=historical_normal.normalized().cross(es.eigenvectors().col(2));
      if(normal.norm()<.5) continue;
      normal.normalize();
      const double residual=normal.dot(world-center);
      if(std::abs(residual)>.5*config.correspondence_radius) continue;
      Constraint c;
      c.point_body=b.point; c.normal_world=normal; c.center_world=center;
      c.residual=residual; c.first=b.first; c.second=b.second;
      c.jacobian.head<3>()=-normal.transpose()*R*skew(b.point);
      c.jacobian.tail<3>()=normal.transpose();
      c.information=reliability({b.first,b.confidence})/(config.boundary_sigma*config.boundary_sigma);
      c.information*=std::min(1.,config.huber/std::max(std::abs(residual),1e-9));
      out.push_back(c);
    }
    // Bound correlated evidence by semantic pair AND by frame; more pixels/points
    // must not create arbitrarily confident pose estimates.
    std::map<std::pair<int,int>,double> sums;
    for(const auto &c:out) sums[{c.first,c.second}]+=c.information;
    double total=0.;
    for(auto &c:out) {
      c.information*=std::min(1.,config.max_group_information/sums[{c.first,c.second}]);
      total+=c.information;
    }
    for(auto &c:out) c.information*=std::min(1.,config.max_frame_information/std::max(total,1e-12));
    return out;
  }
  // Rigid frame changes only. Non-rigid pose-graph corrections must clear/rebuild
  // from corrected keyframes, never silently retain a stale world map.
  void rebase(const Mat3 &R,const Vec3 &t) {
    for(auto &f:frames_) for(auto &b:f.observations) {
      b.point=R*b.point+t; b.surface_normal=R*b.surface_normal;
    }
  }
};
} // namespace semantic_frontend
