# Garment Error Analysis: the gap left by handcrafted features

## Scope and evidence

Phase 1 evaluates Fashion-MNIST with flattened pixels and fixed Histogram of
Oriented Gradients (HOG) descriptors. The SVM experiment exports a normalized
ten-class confusion matrix for each representation, and the accompanying
notebook creates a focused matrix for T-shirt/top (0), Pullover (2), Coat (4),
and Shirt (6), plus galleries of actual test-set mistakes. These artifacts make
the qualitative claim auditable: this report refers to the saved matrices and
images rather than treating an aggregate accuracy score as an explanation.

## The shirt dilemma

Shirt is the most visually ambiguous upper-body label. Its silhouette overlaps
with T-shirts, pullovers, and coats at Fashion-MNIST's 28-by-28 resolution:
all can have a torso-shaped central mass, sleeves, and a low-contrast neckline.
The relevant errors are directional, so the notebook inspects both **true Shirt
→ predicted T-shirt/top** and **true Pullover → predicted Coat** examples.
Those image grids reveal why a confusion entry is not merely a numerical flaw:
the source images often lack enough resolved detail to reliably distinguish a
collar, placket, knit texture, or outerwear layering.

The focused matrices use row normalization against the complete prediction row.
Consequently, an off-diagonal percentage still means “share of this true class
misclassified as that predicted class,” instead of an inflated percentage after
discarding footwear and accessory predictions.

## Why the Phase 1 representations plateau

### Flattened pixels

Flattening converts a 28-by-28 garment into 784 independent positions. A
linear classifier can attach a weight to each position, but it has no built-in
notion that a collar moved slightly upward is the same cue, that two nearby
pixels form a seam, or that left and right sleeve patterns should be composed.
Small translations, pose variation, and contrast changes therefore alter many
coordinates even when the garment category is unchanged.

### HOG descriptors

HOG improves the representation by summarizing local edge orientations, which
helps with distinct global silhouettes such as trousers, bags, sandals, and
boots. However, it fixes the useful edge vocabulary before seeing the task.
It cannot learn that a particular cluster of fine edges around the neckline is
more diagnostic than a similar edge pattern elsewhere, nor can it integrate
button rows, fabric weave, collar shape, and sleeve geometry into a semantic
concept of a shirt. Pooling these gradients into cells also removes precisely
the small spatial cues needed for fine-grained upper-body distinctions.

### Linear decision boundaries

The Linear SVM seeks large-margin separating hyperplanes in the chosen feature
space. That is an appropriate, efficient baseline, but its capacity is bounded
by the information preserved by the input representation. If Shirt and
T-shirt/top share overlapping pixel or HOG descriptions, changing only the
margin penalty `C` cannot create missing discriminative evidence.

## Bridge to Phase 2: learned spatial features

A convolutional neural network is the appropriate next comparison because it
learns feature extractors directly from image neighborhoods. Convolutions share
detectors across locations; stacked layers can combine edges into textures,
collars, sleeves, and garment parts; and pooling supplies modest translation
tolerance. The Phase 2 evaluation should retain the same held-out test split,
report the same per-class recall and confusion matrices, and directly compare
the upper-body submatrix with these AML baselines. A targeted improvement in
Shirt recall and reduced Shirt-to-T-shirt/Pullover confusion would be stronger
evidence than accuracy alone that learned hierarchical spatial features close
the identified gap.
