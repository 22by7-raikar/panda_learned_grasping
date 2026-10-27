# Design notes

Why the pipeline is built the way it is, and the follow-up questions each choice tends to raise.

## Perception

**Why look at the top surface only, and why a 2 cm layer?**
With a single top-down view, the opposite side faces of an object are mostly hidden, so 3D
antipodal sampling has little to work with. The top-surface outline is fully visible and
describes what a top-down parallel gripper actually closes on. The 2 cm layer (from Richtsfeld
& Vincze, as used in the lab's study) stops lower parts from widening the outline. Examples are
a mug's handle base, or the sides of tall objects that are visible in perspective.

**Why Elliptical Fourier Descriptors?**
The raw outline from a depth camera is pixelated and noisy. Normals computed by finite
differences on it jump around by tens of degrees, and the friction-cone test is a test on
normals. A truncated Fourier series is a low-pass filter on the closed curve. It also gives
tangents, normals and curvature analytically from the series derivatives. The harmonic count
trades smoothing against fidelity: about 20 keeps a clamp's neck, while 5 would round the
clamp into a blob.

**What does the friction-cone test do physically?**
A two-finger grasp with Coulomb friction is force-closure in the plane when the line between
the contacts lies inside both friction cones. The fingers then squeeze along that line, and
friction can resist any small disturbance. With μ = 0.4 the cone half-angle is atan(0.4) ≈ 21.8°.
The score prefers pairs well inside the cones, close to the centroid (less torque under gravity
and shaking), and on low-curvature parts of the outline (corners are unstable contacts).

**Why check that the contact line stays inside the outline?**
Without it, the search happily pinches across a gap: two clamp arms, or the hole of a handle.
Both contacts look antipodal, but the fingers would close on air.

**Why sweep the gripper along the approach in the collision check?**
A grasp that is collision-free at its final pose can still hit a taller neighbour on the way
down. Checking the fingers and palm extended 10 cm back along the approach covers that sweep.
That is only valid if the robot really moves in a straight line, which is why the approach is
a RelaxedIK line rather than an OMPL path.

**Why did outlier removal need loosening (stddev 1.0 → 2.0)?**
In simulation, with a 4 mm voxel grid and 1.0σ, statistical outlier removal also stripped about
3 mm from every object edge, because edge points have sparser neighbourhoods. A 4 cm block
measured 3.2 cm. That error goes straight into the grasp width, and it is the kind of detail an
interviewer asks about.

## Learned detector

**Why a two-stage detector (Faster R-CNN) instead of YOLO?**
- **Angle is classified per region.** The proposal network only answers "is there a graspable
  region here", which is orientation-independent. The second stage then classifies the angle
  (19 bins) from each region's pooled features. That is a cleaner split than predicting
  axis-aligned boxes and angles in one shot.
- **Many grasps per object.** One object has several valid grasps at different angles in nearly
  the same place. Per-class NMS, where the class is the angle, keeps them. Class-agnostic NMS in
  single-stage detectors tends to collapse them.
- **Latency isn't the bottleneck.** The robot perceives once and then spends seconds planning
  and moving.
- **It was the published baseline.** Chu et al. had a reproducible Cornell result, which made it
  a fair learned baseline for an analytic-versus-learned comparison.
- **What I'd do today.** For closed-loop grasping, where latency matters, I'd use a pixel-wise
  network such as GG-CNN or GR-ConvNet. For cluttered scenes I'd move to 6-DoF methods such as
  Contact-GraspNet.

**Why RGD instead of RGB-D?**
It keeps a 3-channel input, so ImageNet-pretrained weights transfer directly, which matters
with Cornell's ~885 images. The paper notes a cost: the colour channels make it sensitive to
textured backgrounds and glare. That is where the ResNet model lost the most in the lab's
results.

**How does a 2D rectangle become a robot grasp?**
Take the rectangle centre at the object's top depth (10th percentile inside the rectangle).
Lower it by up to 4 cm, but no lower than the table, which is the 90th percentile of the ring
around the rectangle. The approach runs along the camera's optical axis and the jaw direction
follows the rectangle angle. The width is the pixel width × depth / fx. These are 4-DoF grasps,
correct only because the camera looks straight down.

## Motion planning

**Why two planners (RRTConnect and PRM)?**
RRTConnect is single-query and fast for one-off transits. PRM builds a roadmap that can be
reused across queries in a static scene. The benchmark logs planning time and joint path length
for both.

**What is RelaxedIK doing differently from KDL or TRAC-IK?**
It does not solve for an exact pose. It minimises a weighted sum of "groove" losses:
`-exp(-(x-s)²/2c²) + r(x-s)⁴`, one each for position error, orientation error, joint velocity,
acceleration and jerk, plus joint-limit avoidance. The Gaussian part gives a sharp optimum, and
the quartic keeps a gradient far from it, so terms with different units can be summed. Because
the motion terms pull toward the previous solution, a stream of nearby targets yields a smooth
joint trajectory without IK branch flips.

**What broke, and how was it fixed?**
Near the edge of the workspace (the arm almost straight, for example the 0.8 m scan height),
the motion terms made the solver lag behind each target. It was 1.7 mm off after five
iterations, so the tracker failed. The fix: when the tracked solution has not converged, finish
with a pose-only solve seeded from the current solution. It stays on the same branch, and the
joint-step check still guards continuity. A second bug: the projected BFGS stalled when a joint
sat on its limit, because clamping silently ate the step. Holding active-bound joints fixed for
that step, plus a steepest-descent fallback, together with more iterations and random restarts,
raised the success rate on random reachable goals from 64% to about 93%.

**Which IK + OMPL combination is better?**
OMPL for the transit, and RelaxedIK for the straight approach and lift. See the README for the
argument, and the `approach_max_line_deviation` column of the benchmark CSV for the measurement.

## Benchmark

**Why the Bekiroglu protocol?**
It suits single-object, top-down grasping. The placements are fixed and repeatable (r and θ
around a centre), and the staged score (lift, yaw, shake) separates "picked it up" from "picked
it up stably". GSS is binary; GS gives partial credit. Running the same protocol with other
cameras, lighting and grippers (the paper's 5040 trials) shows which factor a failure comes from.

**What did the lab's study find that this pipeline should be checked against?**
- The analytic methods (Top Surface, Mask-based) scored above 0.8 in most real configurations.
  The learned ones, which were pretrained only, varied a lot with object, background and
  lighting.
- RealSense depth was noisier at 340 lux than at 60 lux. Top Surface improved in low light.
- The ZED 2i's learned depth helped the learned models.
- Top Surface struggled with the medium clamp (small) and the J-cup (concave-hull errors on
  circular hollow objects).
- The Robotiq gripper's larger pads tolerated sub-optimal grasps better than the Franka Hand.
