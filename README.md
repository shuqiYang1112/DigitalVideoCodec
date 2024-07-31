## README

#
 Abstract

This project is a digital video codec system. By using a variety of different algorithms, the gains and losses of various trade-offs (various important parameters) are analyzed. Such as throughput, power consumption, cost, programmability, time to market, and application-specific aspects such as quality, target bit rate, latency, and error resilience.

# Description
<p align="center">
<img src="original_picture.png" alt="Image 1" width="50%">
</p>
<p align="center">
  <img src="figure11.jpg" alt="Image 1" width="30%">
  <img src="figure12.jpg" alt="Image 2" width="30%">
  <img src="figure13.jpg" alt="Image 3" width="30%">
</p>

These images are visual examples of video encoding analysis, demonstrating various encoding features and decisions. The left image shows the selection of a P-frame or I-frame with variable block size decisions overlaid. The middle image depicts the support for multiple reference frames, with each block marked by a color or other indicator showing the index of the reference frame used. The right image displays a map of motion vectors (MVs) overlaid on an arbitrary P-frame, using innovative representations to visualize the direction and magnitude of the motion vectors. These images help to deeply understand and analyze key technologies in the video encoding process.

<p align="center">
  <img src="reconstructed_y_only_encoder.gif" alt="reconstructed_y_only_encoder GIF" width="45%">
</p>

<div style="text-align: center;">
 Figure 2: Six Frames of Reconstructed Y Component
</div>

After decoding, it will generate a reconsturcted YUV file.  This figure shows 6 frames of y component from YUV file .




