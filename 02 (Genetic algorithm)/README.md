<html>
  <head>
    <h2>Genetic algorithm</h2>
  </head>
  <body>
    <h3>Device information:</h3>
    <p><b>CPU</b>: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz</p>
    <p><b>GPU</b>: NVIDIA GeForce GTX 950</p>
    <p>
      <h3>Results:</h3>
      <p><i>Source.cpp</i> - implementation on the CPU (C/C++).</p>
      <p><i>Source.cu</i> - implementation on the GPU (CUDA C).</p>
      <p><i>Generate and plot data.ipynb</i> - visualization (Python).</p>
      <p><i>Output.txt</i> - coefficients of the polynomial in ascending order of degree (starting from 0).</p>
      <p><i>InputX.txt</i> - a set of points by x. The first element is the number of points.</p>
      <p><i>InputY.txt</i> - a set of points by y.</p>
      <table border = "0">
        <caption><h4>Execution time in seconds. 500 points; 5000 epochs.</h4></caption>
        <tr>
          <th rowspan = "2">Population size</th>
          <th colspan = "2">Device</th>
          <th rowspan = "2">CPU time / GPU time</th>
        </tr>
        <tr>
          <td>CPU</td>
          <td>GPU</td>
        </tr>
        <tr>
          <td>1000</td>
          <td>389.787139</td>
          <td>53.522987</td>
          <td>7.2826</td>
        </tr>
        <tr>
          <td>1500</td>
          <td>583.589690</td>
          <td>57.432390</td>
          <td>10.1613</td>
        </tr>
        <tr>
          <td>2000</td>
          <td>778.694910</td>
          <td>83.149958</td>
          <td>9.3649</td>
        </tr>
      </table>
      <p>Approximation function:</p>
      <p><img src=https://latex.codecogs.com/gif.latex?f(x)=0.00000349689196911344*x^0+0.00002875808603013037*x^1+0.00004845544914173874*x^2+0.00000006302847487254*x^3-0.00000000020273538819*x^4.></p>
      <p>Error = 625.774293.</p>
      <h4>Graphic:</h4>
      <p><img src=https://github.com/ChosenByFate/High-performance-computing/blob/master/02%20(Genetic%20algorithm)/Graphics/500%20points.png></p>
      <table border = "0">
        <caption><h4>Execution time in seconds. 1000 points; 2000 epochs.</h4></caption>
        <tr>
          <th rowspan = "2">Population size</th>
          <th colspan = "2">Device</th>
          <th rowspan = "2">CPU time / GPU time</th>
        </tr>
        <tr>
          <td>CPU</td>
          <td>GPU</td>
        </tr>
        <tr>
          <td>1000</td>
          <td>310.074193</td>
          <td>42.444645</td>
          <td>7.3054</td>
        </tr>
        <tr>
          <td>1500</td>
          <td>465.580122</td>
          <td>45.246402</td>
          <td>10.2899</td>
        </tr>
        <tr>
          <td>2000</td>
          <td>657.520273</td>
          <td>65.751609</td>
          <td>10.0001</td>
        </tr>
      </table>
      <p>Approximation function:</p>
      <p><img src=https://latex.codecogs.com/gif.latex?f(x)=12.52614693442646931487*x^0+1.17959173032063602271*x^1-0.00076556205785166640*x^2-0.00000000000000002603*x^3+0.00000000000000003479*x^4.></p>
      <p>Error = 478070.3998.</p>
      <h4>Graphic:</h4>
      <p><img src=https://github.com/ChosenByFate/High-performance-computing/blob/master/02%20(Genetic%20algorithm)/Graphics/1000%20points.png></p>
    </p>
  </body>
  <footer>
    <h3>Log</h3>
    <p>
      GPU

Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1000
Mean of mutation: 0
Variance of mutation: 0.0000001
Number of epochs: 5000

Epoch 4999. Lowest error = 672.936484
Time on GPU = 53.522987 seconds.
0.00000274733459733631 * x^0 + 0.00002153288165835953 * x^1 + 0.00003515833160137862 * x^2 + 0.00000014911428797044 * x^3 + -0.00000000032907787697 * x^4
0.00000274733459733631  0.00002153288165835953  0.00003515833160137862  0.00000014911428797044  -0.00000000032907787697



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1500
Mean of mutation: 0
Variance of mutation: 0.0000001
Number of epochs: 5000

Epoch 4999. Lowest error = 655.329756
Time on GPU = 57.432390 seconds.
0.00000280493496763024 * x^0 + 0.00001407022965044323 * x^1 + 0.00003964033319921679 * x^2 + 0.00000011922445791448 * x^3 + -0.00000000028479681751 * x^4
0.00000280493496763024  0.00001407022965044323  0.00003964033319921679  0.00000011922445791448  -0.00000000028479681751



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 2000
Mean of mutation: 0
Variance of mutation: 0.0000001
Number of epochs: 5000

Epoch 4999. Lowest error = 650.167995
Time on GPU = 83.149958 seconds.
0.00000589887748079164 * x^0 + 0.00001935314572469478 * x^1 + 0.00004089646920394863 * x^2 + 0.00000011056122611413 * x^3 + -0.00000000027110051283 * x^4
0.00000589887748079164  0.00001935314572469478  0.00004089646920394863  0.00000011056122611413  -0.00000000027110051283








CPU

Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1000
Mean of mutation: 0
Variance of mutation: 0.0000001
Number of epochs: 5000

Epoch 4999. Lowest error = 634.985651
Time on CPU = 389.787139 seconds.
0.00000276613223953195 * x^0 + 0.00002985453271375713 * x^1 + 0.00004680635943411066 * x^2 + 0.00000008099807383425 * x^3 + -0.00000000023531579176 * x^4
0.00000276613223953195  0.00002985453271375713  0.00004680635943411066  0.00000008099807383425  -0.00000000023531579176



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1500
Mean of mutation: 0
Variance of mutation: 0.0000001
Number of epochs: 5000

Epoch 4999. Lowest error = 631.919238
Time on CPU = 583.589690 seconds.
0.00000406148502447997 * x^0 + 0.00003062343649821476 * x^1 + 0.00004670170351193428 * x^2 + 0.00000007601567203284 * x^3 + -0.00000000022311427439 * x^4
0.00000406148502447997  0.00003062343649821476  0.00004670170351193428  0.00000007601567203284  -0.00000000022311427439



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 2000
Mean of mutation: 0
Variance of mutation: 0.0000001
Number of epochs: 5000

Epoch 4999. Lowest error = 625.774293
Time on CPU = 778.694910 seconds.
0.00000349689196911344 * x^0 + 0.00002875808603013037 * x^1 + 0.00004845544914173874 * x^2 + 0.00000006302847487254 * x^3 + -0.00000000020273538819 * x^4
0.00000349689196911344  0.00002875808603013037  0.00004845544914173874  0.00000006302847487254  -0.00000000020273538819





GPU

Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1000
Mean of mutation: 0
Variance of mutation: 0.1
Number of epochs: 2000

Epoch 1999. Lowest error = 557022.987571
Time on GPU = 42.444645 seconds.
5.85062999127646943265 * x^0 + 1.17169187140058372876 * x^1 + -0.00074256580663001890 * x^2 + 0.00000000000000000000 * x^3 + 0.00000000000000000000 * x^4
5.85062999127646943265  1.17169187140058372876  -0.00074256580663001890 0.00000000000000000000  0.00000000000000000000



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1500
Mean of mutation: 0
Variance of mutation: 0.1
Number of epochs: 2000

Epoch 1999. Lowest error = 545300.366631
Time on GPU = 45.246402 seconds.
5.42146077571246109983 * x^0 + 1.21460235046193343322 * x^1 + -0.00079710382578884634 * x^2 + 0.00000000000000000000 * x^3 + 0.00000000000000000000 * x^4
5.42146077571246109983  1.21460235046193343322  -0.00079710382578884634 0.00000000000000000000  0.00000000000000000000



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 2000
Mean of mutation: 0
Variance of mutation: 0.1
Number of epochs: 2000

Epoch 1999. Lowest error = 533184.349481
Time on GPU = 65.751609 seconds.
6.69006096033317376737 * x^0 + 1.20417445735178807453 * x^1 + -0.00078291331698241492 * x^2 + 0.00000000000000000000 * x^3 + 0.00000000000000000000 * x^4
6.69006096033317376737  1.20417445735178807453  -0.00078291331698241492 0.00000000000000000000  0.00000000000000000000








CPU

Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1000
Mean of mutation: 0
Variance of mutation: 0.1
Number of epochs: 2000

Epoch 1999. Lowest error = 501095.770565
Time on CPU = 310.074193 seconds.
12.08756125745715337416 * x^0 + 1.20644699633421725871 * x^1 + -0.00080037531480216327 * x^2 + -0.00000000000000001405 * x^3 + 0.00000000000000001890 * x^4
12.08756125745715337416 1.20644699633421725871  -0.00080037531480216327 -0.00000000000000001405 0.00000000000000001890



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 1500
Mean of mutation: 0
Variance of mutation: 0.1
Number of epochs: 2000

Epoch 1999. Lowest error = 484162.006269
Time on CPU = 465.580122 seconds.
12.65631876772098785011 * x^0 + 1.19299631761060886959 * x^1 + -0.00078336610495267499 * x^2 + -0.00000000000000000461 * x^3 + 0.00000000000000004906 * x^4
12.65631876772098785011 1.19299631761060886959  -0.00078336610495267499 -0.00000000000000000461 0.00000000000000004906



Points from files (1 - YES, 0 - NO): 1
Number of individuals (1000 - 2000): 2000
Mean of mutation: 0
Variance of mutation: 0.1
Number of epochs: 2000

Epoch 1999. Lowest error = 478070.399800
Time on CPU = 657.520273 seconds.
12.52614693442646931487 * x^0 + 1.17959173032063602271 * x^1 + -0.00076556205785166640 * x^2 + -0.00000000000000002603 * x^3 + 0.00000000000000003479 * x^4
12.52614693442646931487 1.17959173032063602271  -0.00076556205785166640 -0.00000000000000002603 0.00000000000000003479
    </p>
  </footer>
</html>
