#!/usr/bin/env bash
# make_rviz.sh OUT RGB_EN CLS_EN SCAN_EN FX FY FZ DIST PITCH YAW PTSIZE
OUT=$1; RGB_EN=$2; CLS_EN=$3; SCAN_EN=$4; FX=$5; FY=$6; FZ=$7; DIST=$8; PITCH=$9; YAW=${10}; PS=${11:-2}
cat > "$OUT" <<RVEOF
Panels:
  - Class: rviz_common/Displays
    Help Height: 0
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Map RGB1
        - /Map Class1
      Splitter Ratio: 0.5
    Tree Height: 760
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 10
      Class: rviz_default_plugins/Grid
      Color: 100; 100; 100
      Enabled: false
      Line Style:
        Line Width: 0.03
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset: {X: 0, Y: 0, Z: 0}
      Plane: XY
      Plane Cell Count: 40
      Reference Frame: <Fixed Frame>
      Value: false
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds: {Max Value: 10, Min Value: -10, Value: true}
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: ${RGB_EN}
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: Map RGB
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): ${PS}
      Size (m): 0.15
      Style: Points
      Topic:
        Depth: 1
        Durability Policy: Transient Local
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /semantic_map
      Use Fixed Frame: true
      Use rainbow: true
      Value: ${RGB_EN}
    - Alpha: 1
      Autocompute Intensity Bounds: false
      Autocompute Value Bounds: {Max Value: 10, Min Value: -10, Value: true}
      Axis: Z
      Channel Name: class
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: ${CLS_EN}
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 15
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: Map Class
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): ${PS}
      Size (m): 0.15
      Style: Points
      Topic:
        Depth: 1
        Durability Policy: Transient Local
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /semantic_map
      Use Fixed Frame: true
      Use rainbow: true
      Value: ${CLS_EN}
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds: {Max Value: 10, Min Value: -10, Value: true}
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 30; 30
      Color Transformer: FlatColor
      Decay Time: 0
      Enabled: ${SCAN_EN}
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: Live Scan
      Position Transformer: XYZ
      Selectable: false
      Size (Pixels): 2
      Size (m): 0.2
      Style: Points
      Topic:
        Depth: 1
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /semantic_scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: ${SCAN_EN}
  Enabled: true
  Global Options:
    Background Color: 20; 20; 24
    Fixed Frame: map
    Frame Rate: 20
  Name: root
  Tools:
    - Class: rviz_default_plugins/MoveCamera
  Transformation:
    Current: {Class: rviz_default_plugins/TF}
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: ${DIST}
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.06
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point: {X: ${FX}, Y: ${FY}, Z: ${FZ}}
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: ${PITCH}
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: ${YAW}
    Saved: ~
Window Geometry:
  Displays: {collapsed: false}
  Height: 1040
  Hide Left Dock: false
  Hide Right Dock: true
  Width: 1860
  X: 30
  Y: 20
RVEOF
echo "wrote $OUT"
