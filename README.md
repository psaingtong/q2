# q2
Feature ที่ใช้งาน<br>
1.Amplitude Ratio (ratio)<br>
   feature = |H_RX1| / |H_RX2|<br>
           = sqrt(I1²+Q1²) / sqrt(I2²+Q2²)
           <br>
2.IQ Scatter (iq)<br>
   feature = 2D density map 30×30<br>
             normalize I/Q เป็น unit circle<br>
             plot phase angle ของ H1-H2
             <br><br><br>
ผลจาก iq_scatter.py<br>
<p align="center">
  <img src="iq_scatter_162_134.png" width="45%" /><br>
   
  <img src="iq_scatter_compare_conj.png" width="45%" />
</p>
