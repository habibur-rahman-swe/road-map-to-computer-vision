# Computer Vision Engineer and Researcher Study Path

A schedule-free, from-the-beginning checklist. Work in order where prerequisites matter; take as long as you need. The goal is demonstrable skill: understand the ideas, implement important methods, evaluate them honestly, and communicate reproducible results. Completing a roadmap cannot guarantee a job, degree, or publication, but this path builds the skills and portfolio to pursue them.

## How to Use This Checklist

- [ ] Start with Phase 0 if you are new to programming or mathematics; otherwise test yourself and begin at the first unfamiliar phase.
- [ ] For each topic, learn the concept, implement or use it, and explain it in your own words before checking it off.
- [ ] For theory topics, use this completion test: define the idea, explain why it is useful in vision, work a small example, and name at least one limitation or common mistake.
- [ ] For coding topics, use this completion test: write or run a small example, test an edge case, and explain the input/output shapes and assumptions.
- [ ] Keep notes, code, experiment results, and project reports in a Git repository. Record what you do not understand; revisit it after building something.
- [ ] Use free materials first. A free tier, hosted notebook, or dataset can change or have usage limits; every project should have a small CPU-friendly fallback.
- [ ] Never upload private, sensitive, or personally identifying images to a hosted service without permission. Read dataset, model, and software licenses before reuse or publication.
- [ ] Treat research as a process, not a promise of novelty or publication. Search prior work, ask a narrow question, report negative results, and do not claim more than your experiments show.

## Free Study and Build Toolkit

- [ ] Use Python, JupyterLab, Git, GitHub, NumPy, SciPy, Matplotlib, OpenCV, scikit-image, scikit-learn, and PyTorch. These are open-source and can run locally; a GPU is useful but not required to learn the fundamentals.
- [ ] Learn Python with [Python for Everybody](https://www.py4e.com/) and the [official Python tutorial](https://docs.python.org/3/tutorial/).
- [ ] Learn linear algebra visually with [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra); use [MIT OpenCourseWare](https://ocw.mit.edu/) for deeper free course material.
- [ ] Learn calculus with [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) and practice with [Khan Academy](https://www.khanacademy.org/math/calculus-1).
- [ ] Learn statistics with [StatQuest](https://statquest.org/) and [Seeing Theory](https://seeing-theory.brown.edu/).
- [ ] Use [The Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) and the official NumPy, SciPy, Matplotlib, and pandas documentation as references.
- [ ] Study image processing and geometry with [Szeliski's free Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) and the [OpenCV documentation and tutorials](https://docs.opencv.org/).
- [ ] Study deep learning with the free [Dive into Deep Learning](https://d2l.ai/), [Stanford CS231n course materials](https://cs231n.stanford.edu/), and [PyTorch tutorials](https://pytorch.org/tutorials/).
- [ ] Find papers through [arXiv](https://arxiv.org/), [CVF Open Access](https://openaccess.thecvf.com/), and [Google Scholar](https://scholar.google.com/). Check publication versions and related work, not only abstracts or social posts.
- [ ] Find datasets through research papers, dataset maintainers, [Papers with Code](https://paperswithcode.com/datasets), and [Hugging Face Datasets](https://huggingface.co/datasets). Verify licenses, provenance, splits, and consent/usage terms yourself.
- [ ] If local compute is limited, first reduce image size, batch size, model size, and experiment count. Use CPU-sized samples. Optional hosted notebook free tiers, such as [Google Colab](https://colab.research.google.com/) or [Kaggle Notebooks](https://www.kaggle.com/code), have changing availability and limits; do not make your plan depend on paid compute.
- [ ] Keep a `README`, environment/dependency file, data instructions, and exact run commands for every project. Do not commit large datasets, credentials, or copyrighted material.

## Phase 0: Computer and Learning Foundations

### Theory and concepts
- [x] Learn what files, folders, paths, processes, memory, and programs are.
- [ ] Learn bits, bytes, binary numbers, and how file size differs from image dimensions.
- [ ] Learn file paths, extensions, folders, absolute paths, and relative paths.
- [ ] Learn the difference between source code, an interpreter, a program, and a process.
- [ ] Learn variables, assignment, literals, and basic data types.
- [ ] Learn arithmetic, comparison, boolean operators, and operator precedence.
- [ ] Learn strings, formatting, and converting values between types.
- [ ] Learn `if`/`elif`/`else` decisions and boolean conditions.
- [ ] Learn `for` and `while` loops, loop control, and common off-by-one errors.
- [ ] Learn how to define and call functions, pass arguments, return values, and use scope.
- [ ] Learn how to read a traceback, isolate a small failing example, and use print/debugger inspection.
- [ ] Learn how to read an error message, search official documentation, and ask a precise technical question.

### Practical skills
- [ ] Install Python and an editor, or use a browser notebook when installation is not possible.
- [ ] Run a Python script and a notebook; save, reopen, and modify both.
- [ ] Use a terminal to navigate folders, list files, change directories, and run a script.
- [ ] Create and activate a Python virtual environment; install and import one small package.
- [ ] Create a project folder with a README and a notes file.

### Destination project: Image inventory tool
- [ ] Write a script that scans a folder of your own or openly licensed images and reports file names, dimensions, color channels, and file sizes.
- [ ] Handle a missing folder, a corrupt file, and an unsupported file without crashing the whole run.
- [ ] Save a small CSV report and include instructions for running the script.
- [ ] Explain how a digital image becomes rows, columns, and pixel values.

### Free referrals
- [ ] Use the Python for Everybody lessons and the official Python tutorial linked above.

## Phase 1: Mathematics for Vision

Learn the intuition first, then the notation and calculations. You do not need advanced pure mathematics before beginning image projects.

### Linear algebra
- [ ] Distinguish scalars, vectors, matrices, and higher-dimensional arrays by their shapes.
- [ ] Learn vector addition, scalar multiplication, dot products, and geometric interpretation.
- [ ] Learn vector norms and distances, including Euclidean and Manhattan distance.
- [ ] Learn span, linear independence, basis, dimension, and coordinate representation.
- [ ] Learn matrix addition, transpose, multiplication, and identity matrices.
- [ ] Learn matrix-vector multiplication as a transformation of coordinates.
- [ ] Learn systems of linear equations, rank, singularity, and least-squares solutions.
- [ ] Understand determinant and inverse as concepts; know why explicitly computing an inverse is often numerically unwise.
- [ ] Learn linear transformations, composition, change of basis, and projection.
- [ ] Learn 2D/3D rotation matrices and homogeneous coordinates used for translation.
- [ ] Learn eigenvectors/eigenvalues and interpret dominant directions and scaling.
- [ ] Learn SVD as rotation, scaling, and rotation; connect it to low-rank approximation.
- [ ] Learn tensors as arrays with named dimensions such as batch, channel, height, and width.
- [ ] Check tensor shapes through transpose, reshape, indexing, broadcasting, and matrix multiplication.

### Calculus and optimization
- [ ] Read function graphs and identify inputs, outputs, domain, range, and composition.
- [ ] Review limits and continuity enough to understand local change.
- [ ] Learn derivative as slope/rate of change and calculate derivatives of basic functions.
- [ ] Learn partial derivatives by changing one input while holding other inputs fixed.
- [ ] Learn the gradient as the direction and rate of steepest increase.
- [ ] Learn directional derivatives and interpret a gradient on a loss surface.
- [ ] Learn the Jacobian as a matrix of vector-output partial derivatives.
- [ ] Learn the Hessian as second derivatives and curvature information.
- [ ] Apply the chain rule to nested scalar and vector functions.
- [ ] Derive gradients for a tiny linear model and a simple squared-error loss by hand.
- [ ] Learn gradient descent update steps, learning rate, convergence, and overshooting.
- [ ] Compare batch, stochastic, and mini-batch gradient descent.
- [ ] Learn convexity, local minima, saddle points, and why non-convex optimization can still be useful.
- [ ] Approximate a derivative numerically and compare it with an analytical derivative.

### Probability, statistics, and information
- [ ] Learn outcomes, events, complements, unions, intersections, and probability axioms.
- [ ] Learn conditional probability and distinguish it from joint probability.
- [ ] Learn independence and test whether an independence assumption is plausible.
- [ ] Apply Bayes' rule to update a probability after observing evidence.
- [ ] Learn Bernoulli, categorical, Gaussian/normal, and uniform distributions.
- [ ] Learn expectation as a long-run average and calculate it for a small discrete distribution.
- [ ] Learn variance and standard deviation as measures of spread.
- [ ] Learn covariance and correlation; understand that correlation is not causation.
- [ ] Learn sampling, sampling bias, sample size, and why repeated samples vary.
- [ ] Learn likelihood and maximum likelihood estimation using a simple coin or Gaussian example.
- [ ] Learn confidence intervals and what they do and do not claim.
- [ ] Learn hypothesis tests, p-values, multiple comparisons, and why statistical significance is not practical importance.
- [ ] Learn entropy as uncertainty, cross-entropy as a prediction penalty, and KL divergence as a distribution difference.
- [ ] Learn the roles of training, validation, and test sets; keep test data out of model selection.
- [ ] Recognize leakage, duplicate samples, class imbalance, and uncertainty in reported metrics.

### Numerical computing
- [ ] Learn floating-point precision, overflow/underflow, tolerances, and stable calculations.
- [ ] Learn rounding error and why floating-point addition may depend on operation order.
- [ ] Recognize overflow, underflow, `NaN`, and infinity in calculations.
- [ ] Learn absolute and relative tolerances for comparing floating-point values.
- [ ] Recognize ill-conditioned matrices and why small input changes can cause large output changes.
- [ ] Prefer stable library solvers for least squares and linear systems over unnecessary inverse calculations.
- [ ] Learn random seeds, pseudorandomness, and why identical seeds do not ensure identical results on every device or software version.

### Practical exercises
- [ ] Use NumPy to implement vector operations, matrix products, least squares, and SVD on small examples.
- [ ] Plot a 2D vector transformation and show how changing a matrix changes points and shapes.
- [ ] Implement finite-difference gradients and compare them with symbolic or hand-derived gradients.
- [ ] Simulate coin flips, noisy measurements, and a simple Bayesian update; plot the results.
- [ ] Fit a line by least squares and gradient descent; compare the solutions and convergence behavior.

### Destination project: Math visualizer and optimization notebook
- [ ] Build a notebook that visually demonstrates linear transformations, least squares, and gradient descent.
- [ ] Include derivations in plain language, plots, unit tests for key calculations, and examples where numerical error matters.
- [ ] Verify gradients numerically and report any mismatch rather than hiding it.
- [ ] **Theory referrals:** 3Blue1Brown, MIT OpenCourseWare, Khan Academy, StatQuest, and Seeing Theory in the toolkit above.

## Phase 2: Python, Data, and Reproducible Coding

### Python and software foundations
- [ ] Use strings, numbers, booleans, `None`, and type conversion deliberately.
- [ ] Choose lists, tuples, dictionaries, or sets based on ordering, mutability, and lookup needs.
- [ ] Iterate over collections; use comprehensions only when they remain readable.
- [ ] Define functions with clear parameters, return values, and focused responsibilities.
- [ ] Import standard-library and third-party modules; organize reusable code in a module.
- [ ] Read and write text files with explicit encoding; process CSV and JSON data.
- [ ] Use `pathlib` for paths that work across Windows, macOS, and Linux.
- [ ] Raise, catch, and report expected exceptions without hiding programming errors.
- [ ] Use logging for progress and warnings; use type hints and docstrings to clarify public functions.
- [ ] Write unit tests for normal inputs, empty inputs, invalid inputs, and edge cases.
- [ ] Understand classes, instances, attributes, and methods well enough to read common libraries.
- [ ] Create and activate a `venv`, install pinned dependencies, and reproduce imports in a fresh environment.
- [ ] Use Git status, diff, add, and commit to make small, understandable changes.
- [ ] Use branches and remotes; understand merge conflicts and resolve a simple conflict.
- [ ] Use NumPy arrays and inspect `shape`, `dtype`, range, and memory layout.
- [ ] Index, slice, reshape, transpose, concatenate, and broadcast arrays without unintended copies or shape errors.
- [ ] Use vectorized operations and compare them with explicit Python loops.
- [ ] Use a random generator with an explicit seed for controlled experiments.
- [ ] Plot lines, histograms, scatter plots, and image grids with Matplotlib and label axes/units.
- [ ] Use pandas Series/DataFrames for image metadata, filtering, grouping, and basic summaries.
- [ ] Load and save images with Pillow/OpenCV; check RGB versus BGR and channel order.
- [ ] Convert video into frames and preserve frame order and timestamps where needed.
- [ ] Build a data loader that checks labels, batches samples, and handles a bad file visibly.
- [ ] Separate deterministic evaluation preprocessing from random training augmentation.
- [ ] Use basic command-line commands on your operating system; learn Linux shell basics when relevant.

### Practical exercises
- [ ] Reimplement simple loops using NumPy and compare outputs and runtimes.
- [ ] Write tests for array shape, dtype, boundary conditions, and expected numerical values.
- [ ] Use Git history to track a change, inspect a diff, and restore an individual file from a commit only when intended.
- [ ] Build a small data pipeline that validates inputs and logs invalid samples.

### Destination project: Reproducible image dataset explorer
- [ ] Choose a small, public, permissively usable dataset or a collection you created yourself; record source and license.
- [ ] Write a loader that checks labels, image readability, dimensions, duplicates, and class counts.
- [ ] Visualize representative examples and distributions; note suspicious duplicates, imbalance, and possible split leakage.
- [ ] Provide one command to reproduce the analysis from a clean environment.
- [ ] **Theory/practice referrals:** Python tutorial, Python Data Science Handbook, official NumPy/pandas/Matplotlib docs, Git's free [Pro Git book](https://git-scm.com/book/en/v2), and scikit-learn's [common pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html).

## Phase 3: Digital Images and Classical Image Processing

### Theory and concepts
- [ ] Describe image formation from reflected/emitted light to sensor samples and digital values.
- [ ] Distinguish spatial resolution, bit depth, dynamic range, and file compression.
- [ ] Read image dimensions and interpret grayscale, RGB, RGBA, and channel order.
- [ ] Convert among grayscale, RGB/BGR, HSV, and Lab; explain when a color space helps.
- [ ] Plot and interpret intensity histograms; identify clipping and low contrast.
- [ ] Apply brightness, contrast, normalization, gamma correction, and histogram equalization.
- [ ] Explain nearest-neighbor, bilinear, and bicubic interpolation and choose based on the task.
- [ ] Explain convolution as a local weighted sum; identify kernel size, anchor, and channel behavior.
- [ ] Explain padding, stride, boundary handling, and why output dimensions change.
- [ ] Compare direct convolution with separable filtering for a separable kernel.
- [ ] Add and visualize Gaussian, salt-and-pepper, and Poisson-like noise in a controlled example.
- [ ] Compare Gaussian and median denoising; identify which noise each handles better.
- [ ] Compare bilateral and non-local means denoising; describe quality and speed tradeoffs.
- [ ] Compute image gradients with Sobel/Scharr and distinguish gradient magnitude from direction.
- [ ] Explain Laplacian responses and why noise can make second derivatives unstable.
- [ ] Explain the stages of Canny detection and test how thresholds affect edges.
- [ ] Compare global thresholding, Otsu thresholding, and adaptive thresholding under uneven lighting.
- [ ] Label connected components and compute area, centroid, and bounding box.
- [ ] Explain erosion and dilation through their effect on foreground regions.
- [ ] Use opening to remove small foreground noise and closing to fill small gaps; identify when these assumptions fail.
- [ ] Extract contours and calculate moments, perimeter, area, and polygon approximations.
- [ ] Use convex hulls and watershed; explain marker/seed sensitivity.
- [ ] Build Gaussian or Laplacian pyramids and explain how scale changes image detail.
- [ ] Describe the Fourier transform as a representation of spatial frequencies; inspect magnitude and phase at a basic level.
- [ ] Apply a simple frequency-domain filter and compare its output with a spatial filter.
- [ ] Distinguish lossless and lossy compression; explain basic JPEG quantization artifacts.
- [ ] Calculate MSE, PSNR, and SSIM on a small example and explain why none alone proves perceptual quality.

### Practical exercises
- [ ] Implement convolution for a tiny image and kernel with explicit loops; compare with OpenCV or SciPy.
- [ ] Apply filters to images with different noise types and inspect failure cases.
- [ ] Build a threshold-and-morphology pipeline and measure its behavior on varied lighting/backgrounds.
- [ ] Compare image resizing methods and document aliasing, blur, and edge artifacts.
- [ ] Generate a frequency spectrum and explain how low- and high-frequency changes appear in an image.

### Destination project: Document or object inspection tool
- [ ] Pick a genuinely useful, bounded task such as scanning and cleaning your own notes, counting components on a printed page, or measuring a simple object from images.
- [ ] Implement a classical pipeline using color/grayscale conversion, denoising, thresholding or edges, morphology, and connected components/contours.
- [ ] Test on varied lighting, rotation, blur, and backgrounds; label where it fails.
- [ ] Report measurable criteria (for example, count error or detection precision) against manually checked examples.
- [ ] **Theory/practice referrals:** Szeliski's free book; OpenCV tutorials; [scikit-image user guide](https://scikit-image.org/docs/stable/user_guide.html).

## Phase 4: Classical Computer Vision and Geometry

### Image geometry and cameras
- [ ] Name pixel, image, camera, and world coordinate frames and state the direction of each transform.
- [ ] Represent 2D points with homogeneous coordinates and explain scale-equivalent homogeneous points.
- [ ] Compose translation, rotation, and scale transforms in the correct order.
- [ ] Distinguish rigid, similarity, affine, and projective (homography) transformations by preserved properties.
- [ ] Derive the pinhole projection relation between a 3D point and image coordinates.
- [ ] Interpret focal length, principal point, and intrinsic calibration matrix.
- [ ] Interpret camera rotation/translation as extrinsic pose and transform points between frames.
- [ ] Explain radial and tangential lens distortion and identify visible distortion patterns.
- [ ] Capture or use checkerboard images with varied positions and orientations for calibration.
- [ ] Estimate intrinsics/distortion, inspect reprojection error, and undistort a held-out image.
- [ ] Estimate a homography from at least four non-degenerate point correspondences.
- [ ] Explain image registration and distinguish alignment quality from visual plausibility.
- [ ] Stitch overlapping views and identify parallax, exposure, and seam artifacts.

### Features and matching
- [ ] Explain the aperture problem and why corners are more locally identifiable than edges.
- [ ] Compute a Harris response and tune its neighborhood/threshold parameters.
- [ ] Compare Shi-Tomasi and FAST corner detection on texture-rich and texture-poor images.
- [ ] Distinguish keypoint location/scale/orientation from its descriptor vector.
- [ ] Explain SIFT's scale-space, orientation assignment, and descriptor at a high level.
- [ ] Explain ORB's FAST keypoints and binary rotated BRIEF descriptors.
- [ ] Match binary and floating-point descriptors with appropriate distance measures.
- [ ] Apply nearest-neighbor ratio filtering and inspect both accepted and rejected matches.
- [ ] Use RANSAC to reject outliers; explain its sampling, inlier threshold, and randomness.
- [ ] Compare Hough line/circle detection with contour fitting and state when each is appropriate.

### Motion, 3D, and tracking
- [ ] State brightness constancy and small-motion assumptions in optical flow; show examples where they fail.
- [ ] Track selected points with pyramidal Lucas-Kanade and filter tracks using status/error outputs.
- [ ] Compare sparse flow (selected points) with dense flow (a vector per pixel).
- [ ] Explain epipolar lines and why corresponding points lie on them in calibrated/uncalibrated stereo.
- [ ] Distinguish the fundamental matrix from the essential matrix and state their coordinate assumptions.
- [ ] Rectify stereo images and calculate disparity; interpret disparity sign and invalid regions.
- [ ] Relate disparity, focal length, baseline, and depth; identify why small disparity gives uncertain depth.
- [ ] Triangulate a 3D point from calibrated views and inspect reprojection error.
- [ ] State the PnP inputs/outputs and estimate a camera pose from 2D-3D correspondences.
- [ ] Model a moving object's state and uncertainty with a basic Kalman filter.
- [ ] Explain particle filtering as weighted hypotheses and when it can handle non-Gaussian state distributions.
- [ ] Distinguish object detection, frame-to-frame association, and persistent tracking identity.
- [ ] Explain visual odometry versus SLAM, mapping, localization, drift, and loop closure.
- [ ] Compare feature-based and direct visual odometry at a conceptual level.
- [ ] Explain SfM, camera/point variables, bundle-adjustment reprojection objective, and gauge ambiguity at a high level.

### Practical exercises
- [ ] Estimate a homography from matched points; visualize inliers and reprojection error.
- [ ] Calibrate a camera from a checkerboard and test on a separate image.
- [ ] Estimate sparse optical flow on a short video and inspect motion failures.
- [ ] Rectify a stereo pair and visualize a disparity map; explain why occlusions and textureless regions fail.

### Destination project: Geometry-aware panorama or small reconstruction
- [ ] Capture your own overlapping scene images or use a public dataset with a clear license.
- [ ] Detect and match features, use RANSAC to estimate transforms, and stitch a panorama; alternatively, build a small calibrated stereo reconstruction.
- [ ] Compare at least two feature/matching choices or parameter settings.
- [ ] Evaluate alignment with reprojection error or another justified measurement; show difficult examples.
- [ ] Provide reproducible instructions and a short report explaining assumptions and limitations.
- [ ] **Theory/practice referrals:** Szeliski's free book; OpenCV feature, calibration, stereo, and stitching tutorials; [COLMAP documentation](https://colmap.github.io/) for an established SfM implementation.

## Phase 5: Machine Learning Foundations

### Theory and concepts
- [ ] Distinguish supervised labels, unsupervised structure discovery, and self-supervised targets.
- [ ] Define a model, parameter, feature, target, prediction, objective, and learned representation.
- [ ] Split by the true independent unit (for example, person, video, location, or time) rather than always by image.
- [ ] Use a validation set for model choices and keep test data for final evaluation.
- [ ] Explain k-fold cross-validation and when it is unsuitable (such as dependent grouped samples).
- [ ] Identify data leakage from duplicates, preprocessing, target-derived features, or repeated subjects.
- [ ] Explain overfitting, underfitting, model capacity, and regularization.
- [ ] Fit linear regression and interpret coefficients and residuals.
- [ ] Fit logistic regression and interpret probabilities, logits, and decision thresholds.
- [ ] Explain k-nearest neighbors and how feature scaling affects distance.
- [ ] Explain decision trees, random forests, and support-vector machines at the level needed to select a baseline.
- [ ] Compare full-batch, stochastic, and mini-batch gradient descent.
- [ ] Explain momentum and Adam and identify the role of a learning-rate schedule.
- [ ] Calculate squared-error and binary/multiclass cross-entropy on tiny examples.
- [ ] Explain hinge loss, class weighting, and focal loss use cases at a conceptual level.
- [ ] Read a confusion matrix; calculate accuracy, precision, recall, specificity, and F1.
- [ ] Compare ROC-AUC and PR-AUC and choose based on class balance and use case.
- [ ] Explain threshold selection and why a default threshold may not suit the task.
- [ ] Explain calibration and distinguish confidence from correctness.
- [ ] Choose regression metrics such as MAE, RMSE, and $R^2$ based on error costs and data properties.
- [ ] Use bootstrap resampling or repeated runs to estimate result variability.
- [ ] Define a baseline, ablation, controlled comparison, and error analysis before an experiment.

### Practical exercises
- [ ] Train scikit-learn models on a tabular or handcrafted-image-feature dataset.
- [ ] Compare a majority-class baseline, a simple model, and a more complex model using an untouched test set.
- [ ] Use confusion matrices and inspect examples where the model is wrong.
- [ ] Demonstrate leakage with a deliberately broken split, then correct it.

### Destination project: Classical image classifier and error audit
- [ ] Choose a small open dataset and record its source, license, label meaning, and split procedure.
- [ ] Build a simple feature-based baseline (such as color/texture features with a linear model or SVM).
- [ ] Compare performance with a majority-class baseline and report multiple relevant metrics.
- [ ] Group and inspect errors; identify a plausible data or modeling improvement and test it once.
- [ ] Include a model card-style summary describing intended use, limitations, and risks.
- [ ] **Theory/practice referrals:** [An Introduction to Statistical Learning](https://www.statlearning.com/) (free book and materials), scikit-learn documentation, and StatQuest.

## Phase 6: Deep Learning and PyTorch

### Neural network foundations
- [ ] Compute a perceptron weighted sum and threshold output by hand.
- [ ] Build a multilayer perceptron and distinguish parameters from activations.
- [ ] Compare sigmoid, tanh, ReLU, and softmax and identify output-layer use cases.
- [ ] Distinguish logits, probabilities, labels, and predicted class.
- [ ] Trace a forward pass through a small computation graph and track tensor shapes.
- [ ] Derive gradients with the chain rule and implement backpropagation for a tiny network.
- [ ] Compare manual gradients with PyTorch autograd and finite differences.
- [ ] Explain Xavier/Glorot and He initialization in relation to activation/gradient scale.
- [ ] Build a minibatch and explain shuffle order, batch size, and incomplete final batches.
- [ ] Compare batch normalization and layer normalization at a conceptual level.
- [ ] Explain dropout, L1/L2 regularization (weight decay), and early stopping.
- [ ] Diagnose underfitting/overfitting from training and validation curves.
- [ ] Detect exploding/vanishing gradients and use gradient norms or clipping appropriately.
- [ ] Create PyTorch tensors, move them between CPU/GPU, and avoid accidental device mismatches.
- [ ] Use autograd, zero gradients, perform an optimizer step, and distinguish `train()` from `eval()` mode.
- [ ] Create a PyTorch `Dataset` and `DataLoader` that return the expected image and label types/shapes.

### Convolutional networks
- [ ] Calculate output dimensions from input size, kernel, padding, and stride.
- [ ] Explain input channels, output channels, parameter sharing, and receptive-field growth.
- [ ] Explain max and average pooling and their effects on spatial resolution.
- [ ] Explain transposed convolution and interpolation-plus-convolution for upsampling.
- [ ] Explain residual/skip connections and how they support information/gradient flow.
- [ ] Compare standard, depthwise-separable, and dilated convolution by cost and receptive field.
- [ ] Identify the broad design ideas in LeNet, AlexNet, VGG, ResNet, DenseNet, MobileNet, and EfficientNet.
- [ ] Load pretrained weights and apply the model's required resize, normalization, and channel conventions.
- [ ] Use a pretrained network as a frozen feature extractor, then fine-tune selected or all layers.
- [ ] Choose augmentations that reflect plausible image changes and preserve the target label.
- [ ] Save and reload model weights and the metadata needed for inference.
- [ ] Use inference mode and disable gradient tracking during evaluation.
- [ ] Explain mixed precision benefits and numerical/compatibility limitations.
- [ ] Track configuration, loss/metrics, checkpoint selection, and failed experiments in a local log.

### Practical exercises
- [ ] Train a small MLP and CNN on a small dataset; compare learning curves and parameter counts.
- [ ] Overfit a tiny subset deliberately as a check that the training pipeline can learn.
- [ ] Compare a frozen pretrained model with fine-tuning, if compute allows; otherwise use a smaller model or subset.
- [ ] Run an ablation on one choice such as augmentation, input resolution, or weight decay.
- [ ] Save and reload a model and verify predictions match within an appropriate tolerance.

### Destination project: Reproducible image classification study
- [ ] State a narrow question (for example, how one augmentation affects robustness to a specific image corruption).
- [ ] Define dataset splits, baseline, metric, compute budget, and a small number of experiments before training.
- [ ] Train a simple CNN and one justified reference model; keep all test data untouched during choices.
- [ ] Compare results across relevant conditions, inspect errors, and report run-to-run variability if feasible.
- [ ] Publish code, configuration, dataset instructions, plots, and a short report; avoid claiming a new method unless related-work review supports it.
- [ ] **Theory/practice referrals:** Dive into Deep Learning, CS231n, PyTorch tutorials, and [torchvision models](https://pytorch.org/vision/stable/models.html).

## Phase 7: Core Vision Tasks

Learn task definitions and evaluation before selecting architectures. Be able to reproduce a baseline before modifying it.

### Classification and detection
- [ ] Distinguish single-label, multi-label, and hierarchical image classification.
- [ ] Learn top-1/top-k accuracy, class-wise metrics, calibration, and uncertainty limitations for classification.
- [ ] Represent a detection with class, confidence, and box coordinates; convert between common box formats.
- [ ] Calculate intersection-over-union (IoU) for two boxes and explain its threshold role.
- [ ] Explain candidate boxes/anchors and the role of region proposal networks.
- [ ] Explain non-maximum suppression and identify failure cases with overlapping objects.
- [ ] Explain the broad two-stage R-CNN/Faster R-CNN workflow.
- [ ] Explain the broad one-stage YOLO/SSD/RetinaNet workflow and speed/accuracy tradeoffs.
- [ ] Read a precision-recall curve and distinguish average precision from mean average precision.
- [ ] Explain how matching predictions to ground truth and IoU thresholds affect detection scores.

### Segmentation and localization
- [ ] Distinguish semantic segmentation (class per pixel), instance segmentation (object instances), and panoptic segmentation.
- [ ] Learn masks, ignore labels, void regions, and image-to-mask alignment.
- [ ] Explain encoder-decoder networks and the purpose of skip connections in U-Net-like models.
- [ ] Explain FCN, U-Net, DeepLab, and Mask R-CNN at the level of their task-specific design ideas.
- [ ] Calculate pixel accuracy, class-wise IoU/Jaccard, mean IoU, and Dice score on a tiny mask.
- [ ] Explain boundary metrics and why global pixel scores may hide poor small-object boundaries.
- [ ] Handle class imbalance and absent classes explicitly in a segmentation metric implementation.
- [ ] Represent keypoints and joints with coordinates, visibility/confidence, and a skeleton definition.
- [ ] Explain 2D pose estimation outputs and common normalized keypoint/OKS-style evaluation concepts.

### Video and extended tasks
- [ ] Distinguish single-object tracking (follow a designated target) from multi-object tracking (maintain multiple identities).
- [ ] Explain data association, track birth/death, occlusion, and identity switches.
- [ ] Learn tracking-by-detection and how detector errors propagate into tracks.
- [ ] Interpret tracking metrics such as IDF1/HOTA conceptually; check the official benchmark protocol.
- [ ] Sample video frames uniformly or by time and avoid train/test leakage between adjacent frames.
- [ ] Distinguish video classification, temporal action localization, and action recognition.
- [ ] Understand OCR as text detection followed by recognition, with optional layout/reading-order analysis.
- [ ] Evaluate OCR at word/character level and inspect errors from rotation, blur, fonts, and language.
- [ ] Distinguish monocular, stereo, and learned depth estimates; understand scale ambiguity where relevant.
- [ ] Distinguish denoising, deblurring, super-resolution, and inpainting by the information each task must infer.
- [ ] Identify dataset bias, domain shift, corruption robustness, privacy, and safety concerns specific to the application.

### Practical exercises
- [ ] Fine-tune a small pretrained detector or segmenter on a tiny, openly licensed dataset if feasible; otherwise evaluate a pretrained model on CPU.
- [ ] Visualize predictions, ground truth, and failure cases with correct class names and coordinate scaling.
- [ ] Calculate at least one task metric independently on a few hand-checked examples.
- [ ] Test out-of-distribution conditions such as lighting, blur, camera angle, or background changes.

### Destination project: End-to-end vision application
- [ ] Choose one narrow task with a real user or operational need (examples: sorting a particular recyclable item, identifying plant leaf damage, or counting a defined object type).
- [ ] Establish user constraints, data rights, intended use, and unacceptable failure modes before collecting or selecting images.
- [ ] Make a baseline first; then train/evaluate a suitable model using a documented split and task-appropriate metrics.
- [ ] Build a simple local demo that accepts an image and shows the prediction plus uncertainty or a clear failure state.
- [ ] Evaluate on a separate set reflecting real conditions; document errors, limitations, and what would be unsafe to infer.
- [ ] **Theory/practice referrals:** PyTorch/torchvision tutorials, CS231n, task papers in CVF Open Access, and dataset-specific official documentation.

## Phase 8: Research Methods and Advanced Topics

### Reading and research habits
- [ ] Identify the problem statement and claimed contribution from an abstract; then verify both in the paper body.
- [ ] Inspect figures and tables first, then map each claimed result to its experimental evidence.
- [ ] Identify the paper's assumptions, method inputs/outputs, objective, and computational requirements.
- [ ] Record dataset versions, splits, preprocessing, baselines, metrics, seeds, hardware, and missing details.
- [ ] Check whether comparisons use the same data, split, metric, and evaluation protocol.
- [ ] Trace references backward for foundations and forward for follow-up work, criticism, and independent replication.
- [ ] Search for contradictory evidence and failed replications, not only papers that cite a result positively.
- [ ] Distinguish a new method, new dataset, evaluation/analysis, replication, and useful product engineering contribution.
- [ ] Reproduce a baseline before changing it; document implementation and protocol deviations.
- [ ] Cite ideas, code, data, and figures correctly; never present another person's work as your own.
- [ ] Check consent, privacy, dataset/model licenses, sensitive attributes, bias, and human-subject review requirements.
- [ ] Maintain an experiment log that includes null results, failed runs, and post-hoc changes.

### Advanced methods sampler
- [ ] Derive scaled dot-product attention shapes and explain queries, keys, values, and softmax weights.
- [ ] Explain self-attention, multi-head attention, positional encoding, and transformer blocks.
- [ ] Explain ViT patch embedding, class tokens, and data/compute tradeoffs compared with CNNs.
- [ ] Distinguish supervised pretraining, self-supervised pretraining, and task-specific fine-tuning.
- [ ] Explain contrastive objectives, positive/negative pairs, and risks from false negatives or data leakage.
- [ ] Evaluate learned representations with a linear probe or a justified transfer task.
- [ ] Explain image-text embeddings and zero-shot classification in CLIP-like systems.
- [ ] Evaluate retrieval with recall@k or ranking metrics; inspect prompt sensitivity and model/data licensing.
- [ ] Explain autoencoders and the reconstruction objective; inspect latent representations.
- [ ] Explain VAE latent distributions and the reconstruction/KL tradeoff.
- [ ] Explain GAN generator/discriminator roles, mode collapse, and instability.
- [ ] Explain diffusion forward corruption and learned reverse denoising at a conceptual level.
- [ ] Distinguish domain adaptation, domain generalization, few-shot learning, active learning, and weak supervision.
- [ ] Test robustness under specified corruptions or domain shifts and avoid claiming universal robustness.
- [ ] Explain adversarial examples and threat-model dependence; do not treat a single defense as a guarantee.
- [ ] Compare confidence calibration, uncertainty estimates, saliency maps, and explanation methods; state what they cannot prove.
- [ ] Measure latency, memory, and accuracy before claiming a model is efficient.
- [ ] Explain quantization, pruning, and distillation; test the accuracy/size/runtime tradeoff.
- [ ] Study causal inference, geometric deep learning, manifolds, group symmetry/equivariance, or advanced optimization only when relevant to a defined question.

### Destination project: Paper reproduction
- [ ] Select a recent, understandable paper with available code, data, and a clear baseline; verify licenses and resource needs before committing.
- [ ] Write a one-page reproduction protocol before running experiments: claim to reproduce, exact metrics, data split, environment, compute limits, and planned deviations.
- [ ] Reproduce one central result or a small representative subset within available compute.
- [ ] Compare your result with the paper, report variance/error bars when feasible, and investigate discrepancies rather than tuning silently.
- [ ] Publish a reproduction report that clearly distinguishes reproduced facts, deviations, and unverified claims.
- [ ] **Free referrals:** arXiv, CVF Open Access, Papers with Code for discovery (not as a substitute for paper review), PyTorch, and free hosted notebooks only when available.

## Phase 9: Publishable-Quality Research Capstone

No roadmap can promise that a project is novel enough or accepted by a venue. Aim first for a careful, useful, reproducible study. A negative or mixed result can still be valuable when the question and evaluation are sound.

### Select a feasible real-world question
- [ ] Pick one domain you can access ethically and legally: for example, plant health, local biodiversity, road-surface condition, recycling, document accessibility, or a clearly bounded medical-imaging benchmark.
- [ ] Prefer a question that can be answered with public data, self-collected non-sensitive data, or permissioned data; never scrape or publish images without checking rights and privacy.
- [ ] Narrow the question until it identifies a population/domain, method comparison or intervention, outcome metric, and likely limitation.
- [ ] Search papers, theses, workshop proceedings, and benchmark documentation to identify what is already known.
- [ ] Write down what is genuinely different: a carefully measured domain shift, a local dataset with permission, a low-compute comparison, a failure analysis, a replication, or a well-motivated method change.
- [ ] Ask a teacher, researcher, open-source maintainer, or relevant practitioner for feedback where possible; record the feedback and how it changed the plan.

### Design the study
- [ ] Write a short proposal with context, research question, related work, hypothesis (if appropriate), contribution, risks, and a realistic compute budget.
- [ ] Define the unit of analysis, inclusion/exclusion rules, label process, split method, metrics, and statistical comparisons before inspecting test results.
- [ ] Prevent leakage: split by subject, location, source, or time when random image-level splits would place near-duplicates in train and test.
- [ ] Create a simple baseline and a strong available reference; justify why each comparison is relevant.
- [ ] Plan ablations that isolate one factor at a time; choose a small experiment matrix that fits free CPU or limited free GPU access.
- [ ] Check data and model licenses, privacy, permissions, and any human-subject or institutional review requirements before collecting or sharing data.
- [ ] Decide what evidence would disconfirm the hypothesis and what result would make the study inconclusive.

### Run, analyze, and report
- [ ] Freeze the test set and record code version, configuration, package versions, random seeds, hardware, and run commands.
- [ ] Run sanity checks first: label visualization, tiny-subset overfit, baseline metric, and checkpoint reload.
- [ ] Run planned experiments and record failed runs, changes, compute use, and reasons for deviations.
- [ ] Report task-appropriate metrics, uncertainty/variation where feasible, qualitative examples, subgroup/domain performance when justified, and failure cases.
- [ ] Compare against prior work only when datasets, splits, metrics, and evaluation conditions are comparable; otherwise state the mismatch.
- [ ] Discuss data limitations, bias, privacy, deployment risks, misuse, and what conclusions are not supported.
- [ ] Write a paper-style report: title, abstract, introduction, related work, method, data/ethics, experiments, results, limitations, conclusion, and references.
- [ ] Ask someone else to follow the instructions from a clean environment; fix the missing steps they encounter.
- [ ] Share only artifacts you have rights to share. Provide scripts and data acquisition instructions instead of redistributing restricted data.
- [ ] Get feedback from a relevant mentor or community before choosing a workshop, student venue, or journal; verify current scope, fees, and review process to avoid predatory venues.

### Capstone deliverables checklist
- [ ] A narrow, answerable research question and a documented related-work search.
- [ ] A transparent dataset provenance, license, split, and labeling description.
- [ ] A baseline, justified comparison, and limited but informative ablations.
- [ ] Reproducible code, configuration, dependency list, evaluation script, and results table.
- [ ] Qualitative visualizations and failure analysis, including negative or inconclusive findings.
- [ ] A paper-style report, concise project page/README, and a short presentation.
- [ ] An explicit statement of contribution, limitations, ethical considerations, and future work.
- [ ] A claim that matches the evidence; do not describe an application demo as a research contribution without a supported research claim.

## Phase 10: Engineering, Deployment, and Professional Practice

### Engineering skills
- [ ] Structure code into reusable modules; add tests for data handling, preprocessing, model loading, and evaluation.
- [ ] Save configuration and checkpoints; log metrics and errors locally with reproducible run names.
- [ ] Learn model inference, batching, CPU/GPU device handling, latency measurement, and memory measurement.
- [ ] Learn basic APIs or command-line interfaces and package a small demo without exposing private data.
- [ ] Learn Docker only when it solves a concrete reproducibility or deployment problem; it is not required for learning CV.
- [ ] Learn C++/CUDA only if a performance-critical project or target job needs it; do not let advanced tooling block progress on vision fundamentals.
- [ ] Learn data/model versioning concepts; use free local tools or Git metadata before adopting hosted services.
- [ ] Document installation, data access, expected outputs, known limitations, and how to reproduce evaluation.

### Professional and research portfolio
- [ ] Publish a portfolio with 3-5 polished projects across classical vision, deep learning, and one chosen specialty.
- [ ] For each project, show the problem, your contribution, methods, evidence, limitations, and a runnable path; avoid a gallery of unexplained notebooks.
- [ ] Contribute a documentation fix, test, bug report, or small code improvement to an open-source vision project.
- [ ] Practice explaining one project to a technical audience and a non-technical user without overstating its reliability.
- [ ] Keep a CV/resume that separates implemented skills from topics merely studied.
- [ ] For research roles, seek mentorship, research assistant opportunities, open benchmarks, workshops, and collaboration; formal research jobs may require graduate study or equivalent research experience.
- [ ] Revisit fundamentals whenever a project reveals a gap; advanced research continually uses linear algebra, statistics, careful coding, and experimental design.

## Specialization Branches

Complete the shared foundations, then choose one main branch to study deeply. You can explore another branch later.

### 3D vision, robotics, and SLAM
- [ ] Learn coordinate frames, rigid transforms, rotation representations, and pose composition.
- [ ] Learn multiview geometry, camera calibration, epipolar constraints, and triangulation.
- [ ] Learn stereo matching, depth uncertainty, and limitations from occlusion/textureless surfaces.
- [ ] Learn SfM camera/point estimation and bundle-adjustment residuals.
- [ ] Learn visual odometry drift, loop closure, map reuse, and SLAM evaluation concepts.
- [ ] Compare point clouds, meshes, voxels, and implicit/neural 3D representations at a high level.
- [ ] Build a small reconstruction or localization experiment on a public or self-captured, non-sensitive sequence.
- [ ] Report pose/reconstruction quality, failure cases, and calibration/data assumptions.
- [ ] Referrals: Szeliski, OpenCV, COLMAP, and the documentation/papers for the specific SLAM system studied.

### Medical imaging
- [ ] Learn the image formation and data format for the chosen modality (such as X-ray, CT, MRI, ultrasound, or pathology).
- [ ] Learn task labels, annotation uncertainty, class imbalance, and segmentation/detection metrics in the selected clinical problem.
- [ ] Split by patient (and sometimes site/time), never by individual slices when related slices could leak across partitions.
- [ ] Learn de-identification, data access terms, privacy, clinical validation, and domain shift across sites/scanners.
- [ ] Use only properly licensed, de-identified public data and never present a prototype as a clinical diagnostic tool.
- [ ] Build a benchmark study with patient-level evaluation and explicit limitations; seek domain expert guidance.

### Remote sensing and environmental vision
- [ ] Read raster dimensions, georeferencing, coordinate reference systems, and map projections.
- [ ] Distinguish spectral bands, spatial resolution, ground sample distance, and derived indices.
- [ ] Learn annotation formats and tasks such as land-cover classification, object detection, and change detection.
- [ ] Split geographically or temporally to reduce spatial/seasonal leakage.
- [ ] Test transfer across regions, seasons, sensors, and image resolutions.
- [ ] Build a land-cover, vegetation, or change-detection study using openly licensed imagery and documented geography.
- [ ] Report map-scale limitations, where/when the model may fail, and avoid unsupported environmental conclusions.

### Video, action, and tracking
- [ ] Learn frame rate, timestamps, temporal sampling, shot changes, and video compression effects.
- [ ] Learn optical flow and distinguish camera motion from object motion.
- [ ] Learn detector-based tracking, data association, track management, and identity switches.
- [ ] Learn temporal models for action recognition and video classification at a conceptual level.
- [ ] Split by source video/person/session to prevent adjacent-frame leakage.
- [ ] Learn video evaluation protocols and inspect the official metric implementation.
- [ ] Build a small tracker or action-recognition study using licensed video; blur or avoid bystander faces where appropriate.
- [ ] Test occlusion, camera motion, crowded scenes, and frame-rate changes.

### Vision-language and multimodal systems
- [ ] Learn image/text embedding spaces and contrastive image-text pretraining.
- [ ] Learn image-to-text and text-to-image retrieval; calculate recall@k on a small dataset.
- [ ] Distinguish image captioning from visual question answering and grounded prediction.
- [ ] Learn bounding-box/mask grounding and referring-expression evaluation.
- [ ] Learn open-vocabulary classification/detection and sensitivity to prompt phrasing.
- [ ] Check training-data, model, and image licenses and evaluate bias/domain limitations.
- [ ] Build a retrieval or grounding evaluation with a documented prompt/model protocol and licensed images.
- [ ] Inspect hallucinations and incorrect evidence; do not treat fluent output as proof of visual correctness.

### Imaging, restoration, and generative methods
- [ ] Learn forward degradation models for noise, blur, downsampling, and missing pixels.
- [ ] Learn inverse-problem ill-posedness and distinguish recovering evidence from generating plausible detail.
- [ ] Study denoising, deblurring, super-resolution, and inpainting as separate objectives.
- [ ] Compare pixel metrics with perceptual metrics and human inspection; document metric limitations.
- [ ] Learn diffusion/GAN fundamentals and the risk of plausible hallucinated details.
- [ ] Build a controlled corruption/restoration benchmark with known degradation and compare pixel and perceptual metrics.
- [ ] Include a no-op/input baseline and preserve original images for direct comparison.
- [ ] Report hallucinated details and avoid using generated medical/scientific details as evidence.

## Final Readiness Checklist

- [ ] I can explain the math and assumptions behind core image operations, a CNN, and my chosen vision task.
- [ ] I can create a clean dataset split, identify leakage, select appropriate metrics, and interpret failure cases.
- [ ] I can train, evaluate, save, reload, and run a model without relying on unexplained notebook state.
- [ ] I can implement or reproduce a baseline and explain what changed in my experiment.
- [ ] I can read a paper critically, trace related work, and distinguish evidence from speculation.
- [ ] I can make a project reproducible with documentation, licenses, configuration, and evaluation instructions.
- [ ] I can state the limitations, ethical concerns, and intended use of my work.
- [ ] I have completed a substantial project with a clear question and evidence, not just a tutorial clone.
- [ ] I have chosen whether my next step is an engineering portfolio/job search, a mentored research project, graduate study, or deeper work in a specialization.

**A sensible rule throughout:** understand enough to build, build enough to discover what you do not understand, and measure before you claim success.