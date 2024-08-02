# Abstract

This project is a digital video codec system. By using a variety of different algorithms, the gains and losses of various trade-offs (various important parameters) are analyzed. Such as throughput, power consumption, cost, programmability, time to market, and application-specific aspects such as quality, target bit rate, latency, and error resilience.

# Description
<p align="center">
<img src="original_picture.png" alt="Image 1" width="50%">
<p align="center">Figure 1: Original_frame </p> 
</p>
<p align="center">
  <img src="figure1.png" alt="Image 1">
</p>

These images are visual examples of video encoding analysis, demonstrating various encoding features and decisions. The left image shows the selection of a P-frame or I-frame with variable block size decisions overlaid. The middle image depicts the support for multiple reference frames, with each block marked by a color or other indicator showing the index of the reference frame used. The right image displays a map of motion vectors (MVs) overlaid on an arbitrary P-frame, using innovative representations to visualize the direction and magnitude of the motion vectors. These images help to deeply understand and analyze key technologies in the video encoding process.

<<<<<<< Updated upstream

<p align="center">
  <img src="reconstructed_y_only_encoder.gif" alt="Reconstructed Y Component" style="width: 75%;">
  <p align="center">Figure 2: Six frames of Y component</p>
=======
<<<<<<< HEAD
<p align="center">
<img src="reconstructed_y_only_encoder.gif" alt="Image 1" width="45%">
<figcaption>Figure 2: Six Frames of Reconsturcted Y component</figcaption>
>>>>>>> Stashed changes
</p>
=======
<div align="center">
  <video width="45%" controls>
    <source src="reconstructed_y_only_encoder.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption>Figure 2: Six Frames of Reconstructed Y Component</figcaption>
</div>
>>>>>>> 74b4543366fe9470045028a611f7933dd750f41f


After decoding, it will generate a reconsturcted YUV file.  This figure shows 6 frames of Y component from YUV file .




