Begin with **`prepare_model.py`**.

That is the right first step because your CMUNeXt still has many `BatchNorm2d` layers after convolutions, plus residual adds and decoder concatenations. For PTQ with Jacob-style integer inference, you want the **inference graph first**, not the training graph: BN should be folded into the preceding convs, and the integer path should be defined around the post-folded model. The paper explicitly treats BN folding as part of preparing the inference graph, and its integer runtime relies on offline constants such as scales, zero-points, and multiplier/shift terms.  

With your model, that matters everywhere:

* encoder blocks have depthwise conv → GELU → BN inside residual structure,
* decoder fusion blocks have grouped conv / pointwise conv / GELU / BN,
* skip connections use `torch.cat`,
* and the decoder path is UNet-like.  

So the best next move is **not** to jump into full `int_engine.py`.
Instead, do one clean vertical slice:

## Step 1: finish `prepare_model.py`

Goal: produce a **frozen float inference model** that matches your trained checkpoint but is easier to quantize.

Implement these first:

* load the `.pth`
* set `eval()`
* replace decoder upsample with the version your retrained checkpoint actually uses
* fold every eligible `Conv2d + BatchNorm2d`
* assign stable layer names for tracing

For each foldable pair, verify:

* original module output
* folded module output
* max absolute difference is tiny

This is your first checkpoint:
**“folded float CMUNeXt runs and matches original float CMUNeXt.”**

## Step 2: in the same phase, define your quantization boundaries

Before calibration, decide where activations are quantized.

For your model, the natural boundaries are:

* after each conv block output,
* after each GELU output,
* after each residual add,
* after each max-pool,
* after each upsample+conv block output,
* after each concat,
* after each decoder fusion block,
* after final logits.  

This matters because the paper’s PTQ/QAT formulation quantizes activations at places where tensors would be downcasted during inference, including after activations and after bypass/add/concat-style joins. 

## Step 3: then do `calibrate.py`

Once the folded float model is stable, run your BUSI calibration subset through it and save activation ranges.

Use hooks and collect, at minimum:

* min/max per tensor
* shape
* layer name

Save this to JSON.
Do not overcomplicate range selection yet. Plain min/max is enough for the first pass.

This gives you the activation statistics needed for offline quantization parameters, which is exactly the sort of range information the paper uses to derive scale and zero-point.

## Step 4: then do `quantize.py`

Only after calibration should you quantize:

* weights to int8,
* biases to int32 using input-scale × weight-scale,
* activations using your calibration ranges,
* requantization constants using multiplier + shift.

This file should export:

* quantized weights
* quantized biases
* per-layer qparams
* multiplier / shift
* layer metadata

## Step 5: only then implement `int_ops.py`

Do **not** start with the full model. Start with just these operators:

1. `requantize_int`
2. `conv2d_int`
3. `maxpool_int`
4. `concat_int`
5. `add_int`
6. `gelu_lut_int`

That order matters. If requantization is wrong, everything after it will drift. The paper’s integer runtime depends on the offline multiplier
[
M=\frac{S_xS_w}{S_y}
]
being turned into a fixed-point multiplier plus shift, so this is the core arithmetic piece to get right first. 

## Step 6: test a single block before full `int_engine.py`

Your first end-to-end integer test should be only one small block, for example:

* one folded conv block with ReLU, or
* one CMUNeXt sub-block with conv + GELU LUT + BN-folded conv,
* then compare dequantized output against the folded float version.

Do not start with the entire network.

---

# What you should do this week

Here is the most effective order.

### First

Make `prepare_model.py` work completely.

You should be able to run:

* load model
* fold BN
* save folded model description
* verify output match on one image

### Second

Make `calibrate.py` produce a clean JSON file of activation ranges.

### Third

In `quantize.py`, export one fully quantized block:

* int8 weights
* int32 bias
* output qparams
* multiplier/shift

### Fourth

In `int_ops.py`, get **one conv path** working:

* quantized input
* int32 accumulation
* bias add
* requantization
* optional activation

### Fifth

Add the **GELU LUT** op.

Since your model uses GELU repeatedly in both encoder and fusion blocks, this is one of the central custom ops. 

---

# The most important practical advice

Do **not** begin by wiring the whole nine-file pipeline together.

Begin with this exact milestone:

**Milestone 1:**
“Folded float stem block and one CMUNeXt block run correctly.”

**Milestone 2:**
“Calibration JSON is produced for those blocks.”

**Milestone 3:**
“One block runs in integer arithmetic and matches folded float closely.”

**Milestone 4:**
“GELU LUT works and is traceable.”

**Milestone 5:**
“Full encoder runs.”

**Milestone 6:**
“Full CMUNeXt runs.”

That will save you a lot of debugging time.

---

# If you want the single best starting point

Start by implementing this inside `prepare_model.py`:

* `load_model(checkpoint_path)`
* `fold_conv_bn(conv, bn)`
* `replace_foldable_modules(model)`
* `test_fold_equivalence(model, folded_model, sample_tensor)`

Once that passes, move immediately to `calibrate.py`.

If you share your current `prepare_model.py`, I’ll tell you exactly what functions to put in it first.
